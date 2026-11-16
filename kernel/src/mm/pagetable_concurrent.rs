// SPDX-License-Identifier: MIT OR Apache-2.0
//! Opt-in paging controllers. Hardware roots borrow branches from retained
//! owners; only the owners mutate them. Recursive boot roots are rebuilt,
//! never imported into the paging crate.
extern crate alloc;

use alloc::vec::Vec;
use core::ops::Deref;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};

use super::pagetable_native as native;
use crate::address::{Address, PhysAddr, VirtAddr};
use crate::cpu::control_regs::{read_cr3, write_cr3};
#[cfg(not(all(test, not(target_os = "none"))))]
use crate::cpu::tlb::{TlbFlushRange, TlbFlushScope};
use crate::error::SvsmError;
use crate::mm::alloc::{allocate_pages, free_page};
use crate::mm::{
    PGTABLE_LVL3_IDX_PTE_SELFMAP, PGTABLE_LVL3_IDX_SHARED, PageBox, phys_to_virt, virt_to_phys,
};
use crate::types::{PAGE_SIZE, PAGE_SIZE_2M, PageSize};
use crate::utils::MemoryRegion;
pub use native::{
    ENTRY_COUNT, LAUNCH_VMSA_ADDR, MemAccessMode, PTEntry, PTEntryFlags, PTPage, PTWalkAttr,
    PageFrame, PagingMode, make_private_address, max_phys_addr, paging_init, private_pte_mask,
};
use paging::{
    address::{Address as _, PhysAddr as PagingPhys, VirtAddr as PagingVirt},
    arch::x86_64::{FlushScope, PTEntryFlags as Flags, X86_64Paging, X86_64PagingParams},
    pagetable::{KernelPageTable, UserPageTable},
    structs::{
        Huge, Lvl, NoRootEntries, OwnedSubtree, Page, PageLevel, PageSize as PagingSize,
        PagingAllocator, PagingError, PhysFrame, Regular, SizeLevel2,
    },
    tlb::MayNeedFlush,
};

static ACYCLIC_ACTIVE: AtomicBool = AtomicBool::new(false);
#[cfg(target_os = "none")]
static PAGING_RUNTIME_READY: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "none")]
fn paging_operation_guard() -> Option<crate::cpu::TprGuard> {
    // Bootstrap has no other CPUs or software writers and no initialized GS.
    PAGING_RUNTIME_READY
        .load(Ordering::Acquire)
        .then(|| crate::cpu::TprGuard::raise(crate::types::TPR_LOCK))
}

#[cfg(not(target_os = "none"))]
fn paging_operation_guard() -> Option<()> {
    None
}

/// The paging mutex keeps its queued ticket while invoking this wait hook.
/// Only lock-independent invalidations may run here; arbitrary IPIs stay queued.
#[unsafe(no_mangle)]
extern "C" fn paging_lock_wait() {
    #[cfg(target_os = "none")]
    if PAGING_RUNTIME_READY.load(Ordering::Acquire) {
        crate::cpu::ipi::poll_paging_flush_ipis();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Platform;

fn flush(scope: FlushScope, global: bool, all_cpus: bool) {
    #[cfg(all(test, not(target_os = "none")))]
    {
        tests::record(tests::Event::Flush(scope, global, all_cpus));
    }
    #[cfg(not(all(test, not(target_os = "none"))))]
    {
        let range = match scope {
            FlushScope::All => TlbFlushRange::All,
            FlushScope::Range {
                start,
                end,
                level: PageLevel::Level0,
            } => TlbFlushRange::Range {
                region: MemoryRegion::from_addresses(
                    VirtAddr::new(start.bits()),
                    VirtAddr::new(end.bits()),
                ),
                pgsize: PageSize::Regular,
            },
            FlushScope::Range {
                start,
                end,
                level: PageLevel::Level1,
            } => TlbFlushRange::Range {
                region: MemoryRegion::from_addresses(
                    VirtAddr::new(start.bits()),
                    VirtAddr::new(end.bits()),
                ),
                pgsize: PageSize::Huge,
            },
            // SVSM's range interface has no 1 GiB/table-walk granularity.
            FlushScope::Range { .. } => TlbFlushRange::All,
        };
        let scope = TlbFlushScope { global, range };
        if all_cpus {
            scope.flush_all_cpus();
        } else {
            scope.flush_percpu();
        }
    }
}

// SAFETY: masks and supported flags are initialized by SVSM before allocation.
// Flushes retain SVSM's platform dispatch and synchronous IPI completion; IPI
// handlers only invalidate translations and never acquire the content lock.
unsafe impl X86_64PagingParams for Platform {
    fn private_mask() -> usize {
        private_pte_mask()
    }

    fn shared_mask() -> usize {
        native::shared_pte_mask()
    }

    fn supported_flags() -> Flags {
        flags(native::supported_flags(PTEntryFlags::all()))
    }

    fn flush_tlb_global_sync(scope: FlushScope) {
        flush(scope, true, true);
    }

    fn flush_tlb_global_percpu(scope: FlushScope) {
        flush(scope, true, false);
    }

    fn flush_tlb_ignore_global_sync(scope: FlushScope) {
        flush(scope, false, true);
    }

    fn flush_tlb_ignore_global_percpu(scope: FlushScope) {
        flush(scope, false, false);
    }
}

type Arch = X86_64Paging<Platform>;

struct Allocator;

struct TableMetadata {
    shares: AtomicUsize,
}

const _: () = {
    assert!(paging::structs::PAGE_SIZE == PAGE_SIZE);
    assert!(paging::structs::PT_ENTRY_COUNT == ENTRY_COUNT);
    assert!(core::mem::size_of::<TableMetadata>() <= PAGE_SIZE);
    assert!(core::mem::align_of::<TableMetadata>() <= PAGE_SIZE);
};

fn metadata(address: PagingPhys) -> &'static TableMetadata {
    let pointer = phys_to_virt(PhysAddr::new(address.bits())) + PAGE_SIZE;
    // SAFETY: this allocator initializes metadata in the second page of every
    // allocation. Paging's pinned table access also pins this metadata.
    unsafe { &*pointer.as_ptr::<TableMetadata>() }
}

// Each order-1 allocation contains a table followed by its sharing counter.
// Keeping the counter with the frame avoids an allocating global registry.
// SAFETY: SVSM's fixed heap mapping is writable and stable, with inverse
// translations. Allocations are unique and page aligned. Only paging's
// quiesced owners call deallocation; counter users pin the associated frame.
unsafe impl PagingAllocator for Allocator {
    fn paddr_to_vaddr(address: PagingPhys) -> PagingVirt {
        phys_to_virt(PhysAddr::new(address.bits())).bits().into()
    }

    fn vaddr_to_paddr(address: PagingVirt) -> PagingPhys {
        virt_to_phys(VirtAddr::new(address.bits())).bits().into()
    }

    fn allocate_table_page() -> Result<PagingPhys, PagingError> {
        #[cfg(all(test, not(target_os = "none")))]
        tests::allocation_attempt()?;
        let address = allocate_pages(1).map_err(|_| PagingError::AllocFrame)?;
        // SAFETY: the second page is exclusively owned, suitably aligned
        // storage in this fresh order-1 allocation.
        unsafe {
            (address + PAGE_SIZE)
                .as_mut_ptr::<TableMetadata>()
                .write(TableMetadata {
                    shares: AtomicUsize::new(0),
                });
        }
        let physical = virt_to_phys(address);
        #[cfg(all(test, not(target_os = "none")))]
        tests::record(tests::Event::Allocate(physical.bits()));
        Ok(physical.bits().into())
    }

    unsafe fn deallocate_table_page(address: PagingPhys) {
        assert_eq!(Self::share_count(address).load(Ordering::Acquire), 0);
        #[cfg(all(test, not(target_os = "none")))]
        tests::record(tests::Event::Free(address.bits()));
        free_page(phys_to_virt(PhysAddr::new(address.bits())));
    }

    fn share_count(address: PagingPhys) -> &'static AtomicUsize {
        &metadata(address).shares
    }
}

type Owner = KernelPageTable<Arch, Allocator, Lvl<3>>;
type Root = UserPageTable<Arch, Allocator, Lvl<3>, NoRootEntries>;
type FlushToken = MayNeedFlush<paging::arch::x86_64::X86TlbFlushTok<Platform>>;
type PendingLock = crate::locking::SpinLock<FlushToken>;

struct Branch {
    index: usize,
    table: Owner,
    pending: PendingLock,
}

struct SharedBranch {
    references: AtomicUsize,
    branch: Branch,
}

/// A fallibly allocated owner reference. `Arc::new` cannot report heap
/// exhaustion on the pinned stable toolchain; branch creation must do so.
struct BranchRef(NonNull<SharedBranch>);

impl BranchRef {
    fn new(branch: Branch) -> Result<Self, SvsmError> {
        let allocation = PageBox::try_new(SharedBranch {
            references: AtomicUsize::new(1),
            branch,
        })?;
        Ok(Self(NonNull::from(PageBox::leak(allocation))))
    }

    fn ptr_eq(left: &Self, right: &Self) -> bool {
        left.0 == right.0
    }

    fn shared(&self) -> &SharedBranch {
        // SAFETY: this reference owns one count, pinning initialized storage.
        unsafe { self.0.as_ref() }
    }
}

impl Clone for BranchRef {
    fn clone(&self) -> Self {
        self.shared()
            .references
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                count
                    .checked_add(1)
                    .filter(|next| *next <= isize::MAX as usize)
            })
            .expect("Too many branch references");
        Self(self.0)
    }
}

impl Deref for BranchRef {
    type Target = Branch;

    fn deref(&self) -> &Self::Target {
        &self.shared().branch
    }
}

impl core::fmt::Debug for BranchRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.deref().fmt(f)
    }
}

// SAFETY: Branch controllers support shared atomic observations and keyed
// content locking. Reference counts protect their immutable owner metadata;
// no mutable reference to a SharedBranch escapes after publication.
unsafe impl Send for BranchRef {}
// SAFETY: the same shared-controller and atomic-reference-count protocol
// applies when references are accessed from multiple CPUs.
unsafe impl Sync for BranchRef {}

impl Drop for BranchRef {
    fn drop(&mut self) {
        let _guard = paging_operation_guard();
        if self.shared().references.fetch_sub(1, Ordering::Release) == 1 {
            fence(Ordering::Acquire);
            // SAFETY: the last reference exclusively owns the PageBox leaked
            // by new(). Every borrowing root releases its paging share before
            // dropping its owner reference, after hardware exclusion/flush.
            drop(unsafe { PageBox::from_raw(self.0) });
        }
    }
}

impl core::fmt::Debug for Branch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Branch")
            .field("index", &self.index)
            .finish()
    }
}

impl Branch {
    fn allocate(index: usize) -> Result<BranchRef, SvsmError> {
        let _guard = paging_operation_guard();
        assert!(index < ENTRY_COUNT && index != PGTABLE_LVL3_IDX_PTE_SELFMAP);
        let mut table = Owner::new_empty().map_err(mem_error)?;
        let child = Allocator::allocate_zeroed_table_page().map_err(mem_error)?;
        // SAFETY: this is a freshly allocated zeroed level-2 page, exclusively
        // owned and unpublished, not a populated foreign tree.
        let child = unsafe { OwnedSubtree::<Arch, Allocator, Lvl<2>>::from_owned_root(child) };
        table
            .populate_owned(index, child)
            .map_err(|e| mem_error(e.error))?;
        BranchRef::new(Self {
            index,
            table,
            pending: PendingLock::new(MayNeedFlush::none()),
        })
    }

    fn check(&self, address: VirtAddr) {
        assert_eq!(PageTable::index::<3>(address), self.index);
    }

    fn map<S: PagingSize>(
        &self,
        address: VirtAddr,
        physical: PhysAddr,
        attrs: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        let _guard = paging_operation_guard();
        self.check(address);
        if !attrs.contains(PTEntryFlags::PRESENT) || physical.bits() & !0x000f_ffff_ffff_f000 != 0 {
            return Err(SvsmError::Mem);
        }
        let page = page::<S>(address)?;
        let frame = PhysFrame::<S>::from_start_address(physical.bits().into())
            .map_err(|_| SvsmError::Mem)?;
        self.table
            .map(page, frame, flags(attrs), shared)
            .map_err(mem_error)
    }

    fn unmap<S: PagingSize>(&self, address: VirtAddr) -> Option<PTEntry> {
        let _guard = paging_operation_guard();
        self.check(address);
        let (entry, pending) = self
            .table
            .unmap(page::<S>(address).expect("Unaligned unmap"), None)
            .expect("Wrong unmap leaf size");
        self.retain_pending(pending);
        entry.map(|entry| native_entry(entry.raw()))
    }

    fn retain_pending(&self, pending: FlushToken) {
        let _guard = paging_operation_guard();
        let mut current = self.pending.lock();
        *current = current.and(pending);
    }
}

impl Drop for Branch {
    fn drop(&mut self) {
        let pending = {
            let mut guard = self.pending.lock();
            core::mem::replace(&mut *guard, MayNeedFlush::none())
        };
        // SAFETY: this controller is never loaded in hardware. Every borrowing
        // hardware root pins a BranchRef and releases it only after exclusion
        // and a completed distributed flush. Thus the last reference cannot
        // disappear while any stale translation or structure walk survives.
        // Native mutation callers also keep their original before-reuse flushes.
        unsafe { pending.ignore() };
    }
}

fn mem_error(_: PagingError) -> SvsmError {
    SvsmError::Mem
}

fn flags(attrs: PTEntryFlags) -> Flags {
    Flags::from_bits_retain(attrs.bits() as usize)
}

fn page<S: PagingSize>(address: VirtAddr) -> Result<Page<S>, SvsmError> {
    Page::from_start_address(address.bits().into()).map_err(|_| SvsmError::Mem)
}

fn native_entry(raw: usize) -> PTEntry {
    let mut entry = PTEntry::new_zeroed();
    entry.set_unrestricted(
        PhysAddr::new(raw & 0x000f_ffff_ffff_f000),
        PTEntryFlags::from_bits_truncate(raw as u64),
    );
    entry
}

use zerocopy::FromZeros;

/// A hardware root borrowing separately owned branches. Field order matters:
/// drop the root (releasing its borrow counts) before dropping branch owners.
pub struct PageTable {
    root: Root,
    branches: Vec<BranchRef>,
}

impl core::fmt::Debug for PageTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PageTable")
            .field("cr3", &self.cr3_value())
            .field("branches", &self.branches)
            .finish()
    }
}

impl PageTable {
    /// Enable preemption guards and shootdown polling after BSP per-CPU setup.
    ///
    /// # Safety
    /// The BSP's per-CPU keys must be installed. Every AP must install its own
    /// keys before entering paging, and NMI handlers must not re-enter paging.
    /// Before this call, bootstrap must remain single-threaded and IRQ-disabled.
    #[cfg(target_os = "none")]
    pub unsafe fn enable_runtime_locking() {
        PAGING_RUNTIME_READY.store(true, Ordering::Release);
    }

    pub fn allocate_new() -> Result<PageBox<Self>, SvsmError> {
        let _guard = paging_operation_guard();
        PageBox::try_new(Self {
            root: Root::new_empty().map_err(mem_error)?,
            branches: Vec::new(),
        })
    }

    /// # Safety
    /// The caller must establish the new execution mappings and must not drop
    /// this root while it is active on any CPU. All mutation must use this
    /// backend's owner/lock protocol.
    pub unsafe fn load(&self) {
        // SAFETY: the caller establishes the execution and lifetime contract.
        unsafe { write_cr3(self.cr3_value()) };
        ACYCLIC_ACTIVE.store(true, Ordering::Release);
    }

    pub fn cr3_value(&self) -> PhysAddr {
        PhysAddr::new(self.root.root_paddr().bits())
    }

    pub fn index<const L: usize>(address: VirtAddr) -> usize {
        address.to_pgtbl_idx::<L>()
    }

    fn branch(&self, index: usize) -> Option<&BranchRef> {
        self.branches.iter().find(|branch| branch.index == index)
    }

    fn install(&mut self, branch: BranchRef) -> Result<bool, SvsmError> {
        let _guard = paging_operation_guard();
        if let Some(old) = self.branch(branch.index) {
            assert!(
                BranchRef::ptr_eq(old, &branch),
                "Replacing a branch requires an inactive root"
            );
            return Ok(false);
        }
        self.branches.try_reserve(1).map_err(|_| SvsmError::Mem)?;
        // SAFETY: the owner reference below pins the source until this root has
        // released its borrow. Sources never detach top branches while shared.
        // All access uses paging atomics and the built-in content locks.
        unsafe {
            self.root
                .populate_shared(branch.index, &branch.table, |table| table)
        }
        .map_err(mem_error)?;
        self.branches.push(branch);
        Ok(true)
    }

    fn branch_for_mutation(&mut self, address: VirtAddr) -> Result<&Branch, SvsmError> {
        let index = Self::index::<3>(address);
        if self.branch(index).is_none() {
            let branch = Branch::allocate(index)?;
            self.install(branch)?;
        }
        Ok(self.branch(index).unwrap())
    }

    pub fn clone_shared(&self) -> Result<PageBox<Self>, SvsmError> {
        let mut table = Self::allocate_new()?;
        if let Some(branch) = self.branch(PGTABLE_LVL3_IDX_SHARED) {
            table.install(branch.clone())?;
        }
        Ok(table)
    }

    pub fn copy_entry(&mut self, other: &Self, index: usize) {
        if let Some(branch) = other.branch(index) {
            self.install(branch.clone())
                .expect("Failed to copy shared entry");
        }
    }

    pub fn populate_pgtbl_part(&mut self, part: &PageTablePart) -> bool {
        part.branch.as_ref().is_some_and(|branch| {
            self.install(branch.clone())
                .expect("Failed to install VMR branch")
        })
    }

    /// Replace a per-CPU branch in a dormant incoming task root.
    ///
    /// # Safety
    /// This root must be inactive on every CPU. The scheduler must exclusively
    /// claim the task after the previous CPU has switched CR3 away from it.
    /// No hardware walker may restart on this root until the replacement ends.
    pub unsafe fn populate_pgtbl_part_inactive(&mut self, part: &PageTablePart) -> bool {
        let _guard = paging_operation_guard();
        let Some(branch) = &part.branch else {
            return false;
        };
        if self
            .branch(part.index())
            .is_some_and(|old| BranchRef::ptr_eq(old, branch))
        {
            return false;
        }
        #[cfg(target_os = "none")]
        assert_ne!(self.cr3_value(), clean_address(read_cr3().bits()));
        if let Some(position) = self
            .branches
            .iter()
            .position(|old| old.index == part.index())
        {
            // Hardware is excluded by the scheduler's CR3/active handshake.
            // Complete invalidation before releasing cached structure references.
            flush(FlushScope::All, true, true);
            // SAFETY: the caller excludes all hardware users; self is exclusive
            // and the preceding synchronous flush excludes stale cached walks.
            unsafe { self.root.release_shared(part.index()) }.expect("Missing old branch");
            self.branches.remove(position);
        }
        self.install(branch.clone())
            .expect("Failed to replace VMR branch")
    }

    pub fn map_4k(
        &mut self,
        address: VirtAddr,
        physical: PhysAddr,
        attrs: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        self.branch_for_mutation(address)?
            .map::<Regular>(address, physical, attrs, shared)
    }

    pub fn map_2m(
        &mut self,
        address: VirtAddr,
        physical: PhysAddr,
        attrs: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        assert!(address.is_aligned(PAGE_SIZE_2M) && physical.is_aligned(PAGE_SIZE_2M));
        self.branch_for_mutation(address)?
            .map::<Huge>(address, physical, attrs, shared)
    }

    pub fn unmap_4k(&mut self, address: VirtAddr) {
        if let Some(branch) = self.branch(Self::index::<3>(address)) {
            branch.unmap::<Regular>(address);
        }
    }

    pub fn unmap_2m(&mut self, address: VirtAddr) {
        assert!(address.is_aligned(PAGE_SIZE_2M));
        if let Some(branch) = self.branch(Self::index::<3>(address)) {
            branch.unmap::<Huge>(address);
        }
    }

    pub fn phys_addr(&self, address: VirtAddr) -> Result<PhysAddr, SvsmError> {
        self.root
            .phys_addr(address.bits().into())
            .map(|physical| PhysAddr::new(physical.bits()))
            .map_err(mem_error)
    }

    pub fn check_mapping(&self, address: VirtAddr) -> Option<PhysAddr> {
        let mapping = self.root.walk(address.bits().into());
        mapping
            .read()
            .is_present_leaf(mapping.level())
            .then(|| PhysAddr::new(mapping.read().leaf_address(mapping.level()).bits()))
    }

    fn visibility(&self, address: VirtAddr, shared: bool) -> Result<(), SvsmError> {
        let _guard = paging_operation_guard();
        let branch = self
            .branch(Self::index::<3>(address))
            .ok_or(SvsmError::Mem)?;
        let page = page::<Regular>(address)?;
        let pending = if shared {
            branch.table.set_shared(page, true)
        } else {
            branch.table.set_private(page, true)
        }
        .map_err(mem_error)?;
        // Retain the obligation without adding a shootdown while the caller's
        // outer page-table/VMR locks are held. Native callers flush after those
        // guards have been released, before cache/platform transitions/reuse.
        branch.retain_pending(pending);
        Ok(())
    }

    pub fn set_shared_4k(&mut self, address: VirtAddr) -> Result<(), SvsmError> {
        self.visibility(address, true)
    }

    pub fn set_encrypted_4k(&mut self, address: VirtAddr) -> Result<(), SvsmError> {
        self.visibility(address, false)
    }

    pub fn map_region_4k(
        &mut self,
        region: MemoryRegion<VirtAddr>,
        physical: PhysAddr,
        attrs: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        for address in region.iter_pages(PageSize::Regular) {
            self.map_4k(
                address,
                physical + (address - region.start()),
                attrs,
                shared,
            )?;
        }
        Ok(())
    }

    pub fn map_region_2m(
        &mut self,
        region: MemoryRegion<VirtAddr>,
        physical: PhysAddr,
        attrs: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        for address in region.iter_pages(PageSize::Huge) {
            self.map_2m(
                address,
                physical + (address - region.start()),
                attrs,
                shared,
            )?;
        }
        Ok(())
    }

    pub fn unmap_region_4k(&mut self, region: MemoryRegion<VirtAddr>) {
        for address in region.iter_pages(PageSize::Regular) {
            self.unmap_4k(address);
        }
    }

    pub fn unmap_region_2m(&mut self, region: MemoryRegion<VirtAddr>) {
        for address in region.iter_pages(PageSize::Huge) {
            self.unmap_2m(address);
        }
    }

    pub fn map_region(
        &mut self,
        region: MemoryRegion<VirtAddr>,
        physical: PhysAddr,
        attrs: PTEntryFlags,
    ) -> Result<(), SvsmError> {
        let mut address = region.start();
        while address < region.end() {
            let frame = physical + (address - region.start());
            if address.is_aligned(PAGE_SIZE_2M)
                && frame.is_aligned(PAGE_SIZE_2M)
                && region.end() - address >= PAGE_SIZE_2M
                && self.map_2m(address, frame, attrs, false).is_ok()
            {
                address = address + PAGE_SIZE_2M;
            } else {
                self.map_4k(address, frame, attrs, false)?;
                address = address + PAGE_SIZE;
            }
        }
        Ok(())
    }

    pub fn unmap_region(&mut self, region: MemoryRegion<VirtAddr>) {
        let mut address = region.start();
        while address < region.end() {
            let mapping = self.root.walk(address.bits().into());
            match mapping.level() {
                PageLevel::Level0 => {
                    self.unmap_4k(address);
                    address = address + PAGE_SIZE;
                }
                PageLevel::Level1 => {
                    self.unmap_2m(address);
                    address = address + PAGE_SIZE_2M;
                }
                _ => address = address + PAGE_SIZE,
            }
        }
    }

    /// # Safety
    /// The caller must exclude writes to these global, 4 KiB aligned pages,
    /// just as for the native implementation.
    pub unsafe fn make_region_ro_4k(
        &mut self,
        region: MemoryRegion<VirtAddr>,
    ) -> Result<(), SvsmError> {
        let _guard = paging_operation_guard();
        for address in region.iter_pages(PageSize::Regular) {
            let observed = self.root.walk(address.bits().into());
            if !observed.read().present() {
                continue;
            }
            if observed.level() != PageLevel::Level0
                || !observed.read().flags().contains(Flags::GLOBAL)
            {
                return Err(SvsmError::Mem);
            }
            let branch = self
                .branch(Self::index::<3>(address))
                .ok_or(SvsmError::Mem)?;
            branch.retain_pending(
                branch
                    .table
                    .set_flags(
                        page::<Regular>(address)?,
                        flags(PTEntryFlags::data_ro()),
                        true,
                    )
                    .map_err(mem_error)?,
            );
        }
        Ok(())
    }

    /// Read the currently active root, including during early recursive boot.
    pub fn virt_to_frame(address: VirtAddr) -> Option<PageFrame> {
        if !ACYCLIC_ACTIVE.load(Ordering::Acquire) {
            return native::PageTable::virt_to_frame(address);
        }
        let root = clean_address(read_cr3().bits());
        // SAFETY: CR3 identifies the active root, whose tables stay pinned and
        // heap-mapped. Software follows this backend's atomic access protocol.
        unsafe { frame_from_root(phys_to_virt(root), address) }
    }

    /// Rebuild the boot hierarchy without copying its recursive root link.
    /// Original boot table pages remain reserved in the boot heap.
    ///
    /// # Safety
    /// `root` must be the trusted initialized four-level boot hierarchy.
    /// All its table pages must be in SVSM's fixed heap mapping. No other CPU
    /// or software writer may modify it until rebuilding and the CR3 switch end.
    pub unsafe fn rebuild_boot_root(root: VirtAddr) -> Result<PageBox<Self>, SvsmError> {
        let mut table = Self::allocate_new()?;
        for index in 0..ENTRY_COUNT {
            if index == PGTABLE_LVL3_IDX_PTE_SELFMAP {
                continue;
            }
            // SAFETY: the caller supplies an initialized pinned boot root.
            let entry = unsafe { load_entry(root, index) };
            if entry & 1 == 0 {
                continue;
            }
            if entry & (1 << 7) != 0 {
                return Err(SvsmError::Mem);
            }
            let branch = Branch::allocate(index)?;
            let prefix = VirtAddr::new(index << 39);
            // SAFETY: the trusted root's present non-leaf points to a valid
            // heap-mapped level-2 table, with no concurrent software writers.
            unsafe {
                rebuild_level(
                    &branch,
                    phys_to_virt(clean_address(entry)),
                    2,
                    prefix,
                    entry,
                )
            }?;
            table.install(branch)?;
        }
        Ok(table)
    }
}

impl Drop for PageTable {
    fn drop(&mut self) {
        #[cfg(target_os = "none")]
        assert_ne!(
            self.cr3_value(),
            clean_address(read_cr3().bits()),
            "Dropping active page tables"
        );
        // The unsafe load/scheduler contract excludes all CPUs from this root.
        // Invalidate stale structure/leaf caches before root and source Drop.
        flush(FlushScope::All, true, true);
    }
}

fn clean_address(entry: usize) -> PhysAddr {
    PhysAddr::new(entry & 0x000f_ffff_ffff_f000 & !(private_pte_mask() | native::shared_pte_mask()))
}

unsafe fn load_entry(table: VirtAddr, index: usize) -> usize {
    // SAFETY: the caller supplies initialized, pinned, aligned hardware PTE
    // storage. Atomic observations coexist with hardware A/D updates.
    unsafe { (&*table.as_ptr::<AtomicUsize>().add(index)).load(Ordering::Acquire) }
}

unsafe fn frame_from_root(root: VirtAddr, address: VirtAddr) -> Option<PageFrame> {
    let mut table = root;
    for level in (0..=3).rev() {
        let index = (address.bits() >> (12 + 9 * level)) & 511;
        // SAFETY: the root is valid and each preceding entry was a present
        // table pointer pinned in the fixed heap mapping.
        let entry = unsafe { load_entry(table, index) };
        if entry & 1 == 0 {
            return None;
        }
        if level == 0 || (level < 3 && entry & (1 << 7) != 0) {
            let size = 1usize << (12 + 9 * level);
            let frame = (entry & 0x000f_ffff_ffff_f000 & !private_pte_mask()) & !(size - 1);
            let physical = PhysAddr::new(frame + (address.bits() & (size - 1)));
            return Some(match level {
                0 => PageFrame::Size4K(physical),
                1 => PageFrame::Size2M(physical),
                2 => PageFrame::Size1G(physical),
                _ => unreachable!(),
            });
        }
        table = phys_to_virt(clean_address(entry));
    }
    None
}

unsafe fn rebuild_level(
    branch: &Branch,
    table: VirtAddr,
    level: usize,
    prefix: VirtAddr,
    parents: usize,
) -> Result<(), SvsmError> {
    for index in 0..ENTRY_COUNT {
        // SAFETY: the caller supplies a trusted, initialized boot table.
        let mut entry = unsafe { load_entry(table, index) };
        if entry & 1 == 0 {
            continue;
        }
        let address = prefix + (index << (12 + 9 * level));
        // Collapse ancestor access restrictions into leaves. New intermediate
        // entries are permissive; leaf history/global/confidential bits survive.
        entry = (entry & !6) | (entry & parents & 6);
        entry |= parents & (1usize << 63);
        if level > 0 && entry & (1 << 7) == 0 {
            // SAFETY: a present non-leaf in the trusted boot tree points to a
            // heap-mapped table one level below it.
            unsafe {
                rebuild_level(
                    branch,
                    phys_to_virt(clean_address(entry)),
                    level - 1,
                    address,
                    entry,
                )
            }?;
        } else {
            let known = PTEntryFlags::all().bits() as usize | 0x000f_ffff_ffff_f000;
            if entry & !known != 0
                || (level == 0 && entry & (1 << 7) != 0)
                || (level > 0 && entry & ((1usize << (12 + level * 9)) - PAGE_SIZE) != 0)
            {
                // In particular, never silently discard a huge-page PAT bit.
                return Err(SvsmError::Mem);
            }
            let attrs = PTEntryFlags::from_bits_truncate(entry as u64) & !PTEntryFlags::HUGE;
            let shared = entry & native::shared_pte_mask() == native::shared_pte_mask()
                && entry & private_pte_mask() == 0;
            let physical = clean_address(entry);
            match level {
                0 => branch.map::<Regular>(address, physical, attrs, shared)?,
                1 => branch.map::<Huge>(address, physical, attrs, shared)?,
                2 => branch.map::<SizeLevel2>(address, physical, attrs, shared)?,
                _ => return Err(SvsmError::Mem),
            }
        }
    }
    Ok(())
}

/// A VMR-owned branch retained by every hardware root that borrows it.
#[derive(Debug)]
pub struct PageTablePart {
    index: usize,
    branch: Option<BranchRef>,
}

impl PageTablePart {
    pub fn new(start: VirtAddr) -> Self {
        Self {
            index: PageTable::index::<3>(start),
            branch: None,
        }
    }

    pub fn alloc(&mut self) {
        self.get_or_init()
            .expect("Failed to allocate page table branch");
    }

    fn get_or_init(&mut self) -> Result<&BranchRef, SvsmError> {
        if self.branch.is_none() {
            self.branch = Some(Branch::allocate(self.index)?);
        }
        Ok(self.branch.as_ref().unwrap())
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn address(&self) -> Option<PhysAddr> {
        self.branch.as_ref().and_then(|branch| {
            branch
                .table
                .next_table_pa(self.index)
                .map(|address| PhysAddr::new(address.bits()))
        })
    }

    pub fn map_4k(
        &mut self,
        address: VirtAddr,
        physical: PhysAddr,
        attrs: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        assert_eq!(PageTable::index::<3>(address), self.index);
        self.get_or_init()?
            .map::<Regular>(address, physical, attrs, shared)
    }

    pub fn map_2m(
        &mut self,
        address: VirtAddr,
        physical: PhysAddr,
        attrs: PTEntryFlags,
        shared: bool,
    ) -> Result<(), SvsmError> {
        assert_eq!(PageTable::index::<3>(address), self.index);
        assert!(address.is_aligned(PAGE_SIZE_2M) && physical.is_aligned(PAGE_SIZE_2M));
        self.get_or_init()?
            .map::<Huge>(address, physical, attrs, shared)
    }

    pub fn unmap_4k(&mut self, address: VirtAddr) -> Option<PTEntry> {
        assert_eq!(PageTable::index::<3>(address), self.index);
        self.branch
            .as_ref()
            .and_then(|branch| branch.unmap::<Regular>(address))
    }

    pub fn unmap_2m(&mut self, address: VirtAddr) -> Option<PTEntry> {
        assert_eq!(PageTable::index::<3>(address), self.index);
        self.branch
            .as_ref()
            .and_then(|branch| branch.unmap::<Huge>(address))
    }
}

#[cfg(all(test, not(target_os = "none")))]
#[path = "pagetable_concurrent/tests.rs"]
mod tests;
