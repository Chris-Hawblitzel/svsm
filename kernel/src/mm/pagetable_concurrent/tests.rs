// SPDX-License-Identifier: MIT OR Apache-2.0
//! Executable host tests of the selected SVSM backend, using SVSM's allocator.

use super::*;
use crate::locking::SpinLock;
use crate::mm::alloc::{DEFAULT_TEST_MEMORY_SIZE, TestRootMem};
use crate::mm::{PGTABLE_LVL3_IDX_PERCPU, PGTABLE_LVL3_IDX_PERTASK, virt_from_idx};
use core::sync::atomic::AtomicIsize;
use paging::tlb::MayNeedFlush;

#[derive(Clone, Debug)]
pub(super) enum Event {
    Allocate(usize),
    Free(usize),
    Flush(FlushScope, bool, bool),
}

static EVENTS: SpinLock<Vec<Event>> = SpinLock::new(Vec::new());
static FAIL_AFTER: AtomicIsize = AtomicIsize::new(-1);

pub(super) fn record(event: Event) {
    EVENTS.lock().push(event);
}

pub(super) fn allocation_attempt() -> Result<(), PagingError> {
    let remaining = FAIL_AFTER.load(Ordering::Relaxed);
    if remaining == 0 {
        return Err(PagingError::AllocFrame);
    }
    if remaining > 0 {
        FAIL_AFTER.fetch_sub(1, Ordering::Relaxed);
    }
    Ok(())
}

fn setup() -> TestRootMem<'static> {
    let guard = TestRootMem::setup(DEFAULT_TEST_MEMORY_SIZE);
    native::init_backend_test_masks();
    EVENTS.lock().clear();
    FAIL_AFTER.store(-1, Ordering::Relaxed);
    guard
}

fn assert_reclaimed() {
    let events = EVENTS.lock();
    let mut allocated = Vec::new();
    let mut freed = Vec::new();
    for event in events.iter() {
        match event {
            Event::Allocate(address) => allocated.push(*address),
            Event::Free(address) => freed.push(*address),
            Event::Flush(..) => {}
        }
    }
    allocated.sort_unstable();
    freed.sort_unstable();
    assert_eq!(
        allocated, freed,
        "every table allocation must be reclaimed exactly once"
    );
}

fn observed(table: &PageTable, address: VirtAddr) -> paging::structs::PTEntry<Arch> {
    table.root.walk(address.bits().into()).read()
}

#[test]
fn host_wait_hook_and_operation_guard_need_no_per_cpu_or_privileged_state() {
    paging_lock_wait();
    assert!(paging_operation_guard().is_none());
}

#[test]
fn mappings_use_real_svsm_root_indices_and_preserve_flags_and_offsets() {
    let _memory = setup();
    let mut table = PageTable::allocate_new().unwrap();
    assert!(
        table
            .root
            .next_table_pa(PGTABLE_LVL3_IDX_PTE_SELFMAP)
            .is_none()
    );
    for index in [0, PGTABLE_LVL3_IDX_PERTASK, PGTABLE_LVL3_IDX_PERCPU, 511] {
        let address = virt_from_idx(index) + 0x4000;
        table
            .map_4k(address, PhysAddr::new(0x8000), PTEntryFlags::data(), false)
            .unwrap();
        assert_eq!(
            table.phys_addr(address + 0x321).unwrap(),
            PhysAddr::new(0x8321)
        );
        let entry = observed(&table, address);
        assert!(
            entry
                .flags()
                .contains(Flags::GLOBAL | Flags::NX | Flags::WRITABLE)
        );
        assert_eq!(entry.paddr_field() & private_pte_mask(), private_pte_mask());
        assert_eq!(entry.paddr_field() & native::shared_pte_mask(), 0);
    }
    assert!(table.phys_addr(VirtAddr::new(0x3000)).is_err());
    drop(table);
    assert_reclaimed();
}

#[test]
fn visibility_split_preserves_neighbors_and_both_confidentiality_tags() {
    let _memory = setup();
    let mut table = PageTable::allocate_new().unwrap();
    let address = virt_from_idx(511) + PAGE_SIZE_2M;
    table
        .map_2m(
            address,
            PhysAddr::new(0x400000),
            PTEntryFlags::data(),
            false,
        )
        .unwrap();
    table.set_shared_4k(address + PAGE_SIZE).unwrap();
    assert_eq!(
        table.root.walk((address + PAGE_SIZE).bits().into()).level(),
        PageLevel::Level0
    );
    let shared = observed(&table, address + PAGE_SIZE);
    let private = observed(&table, address + 2 * PAGE_SIZE);
    assert!(shared.is_shared());
    assert!(!private.is_shared());
    assert_eq!(shared.paddr_field() & private_pte_mask(), 0);
    assert_eq!(
        shared.paddr_field() & native::shared_pte_mask(),
        native::shared_pte_mask()
    );
    assert!(
        shared
            .flags()
            .contains(Flags::NX | Flags::GLOBAL | Flags::ACCESSED | Flags::DIRTY)
    );
    assert_eq!(
        table.phys_addr(address + PAGE_SIZE + 17).unwrap(),
        PhysAddr::new(0x401011)
    );
    assert_eq!(
        table.phys_addr(address + 2 * PAGE_SIZE).unwrap(),
        PhysAddr::new(0x402000)
    );
    table.set_encrypted_4k(address + PAGE_SIZE).unwrap();
    assert!(!observed(&table, address + PAGE_SIZE).is_shared());
    table.unmap_4k(address + PAGE_SIZE);
    assert!(table.phys_addr(address + PAGE_SIZE).is_err());
    assert!(table.phys_addr(address + 2 * PAGE_SIZE).is_ok());
    assert!(
        EVENTS
            .lock()
            .iter()
            .any(|e| matches!(e, Event::Flush(_, true, true)))
    );
    drop(table);
    assert_reclaimed();
}

#[test]
fn failed_split_leaves_the_original_huge_mapping_unchanged() {
    let _memory = setup();
    let mut table = PageTable::allocate_new().unwrap();
    let address = virt_from_idx(511);
    table
        .map_2m(
            address,
            PhysAddr::new(0x400000),
            PTEntryFlags::data(),
            false,
        )
        .unwrap();
    let before = observed(&table, address).raw();
    FAIL_AFTER.store(0, Ordering::Relaxed);
    assert!(table.set_shared_4k(address + PAGE_SIZE).is_err());
    FAIL_AFTER.store(-1, Ordering::Relaxed);
    assert_eq!(observed(&table, address).raw(), before);
    assert_eq!(
        table.root.walk(address.bits().into()).level(),
        PageLevel::Level1
    );
    drop(table);
    assert_reclaimed();
}

#[test]
fn concurrent_borrowers_split_once_and_update_distinct_leaves() {
    extern crate std;
    let _memory = setup();
    let mut source = PageTable::allocate_new().unwrap();
    let address = virt_from_idx(511);
    source
        .map_2m(
            address,
            PhysAddr::new(0x400000),
            PTEntryFlags::data(),
            false,
        )
        .unwrap();
    let mut first = source.clone_shared().unwrap();
    let mut second = source.clone_shared().unwrap();
    std::thread::scope(|scope| {
        scope.spawn(|| first.set_shared_4k(address + PAGE_SIZE).unwrap());
        scope.spawn(|| second.set_shared_4k(address + 2 * PAGE_SIZE).unwrap());
    });
    for offset in [PAGE_SIZE, 2 * PAGE_SIZE] {
        assert!(observed(&source, address + offset).is_shared());
        assert_eq!(
            source.phys_addr(address + offset).unwrap(),
            PhysAddr::new(0x400000 + offset)
        );
    }
    assert!(!observed(&source, address + 3 * PAGE_SIZE).is_shared());
    drop((first, second, source));
    assert_reclaimed();
}

#[test]
fn shared_kernel_mutation_uses_owner_and_survives_source_root_drop() {
    let _memory = setup();
    let mut source = PageTable::allocate_new().unwrap();
    let address = virt_from_idx(511) + 0x4000;
    source
        .map_4k(address, PhysAddr::new(0x8000), PTEntryFlags::data(), false)
        .unwrap();
    let mut first = source.clone_shared().unwrap();
    let second = source.clone_shared().unwrap();
    let child = source.root.next_table_pa(511).unwrap();
    assert_eq!(Allocator::share_count(child).load(Ordering::Acquire), 3);
    drop(source);
    assert_eq!(Allocator::share_count(child).load(Ordering::Acquire), 2);
    first.set_shared_4k(address).unwrap();
    assert!(observed(&second, address).is_shared());
    first.unmap_4k(address);
    assert!(second.phys_addr(address).is_err());
    drop(first);
    assert_eq!(Allocator::share_count(child).load(Ordering::Acquire), 1);
    drop(second);
    assert_reclaimed();
}

#[test]
fn vmr_owner_is_retained_until_last_root_and_unmap_returns_original_leaf() {
    let _memory = setup();
    let address = virt_from_idx(PGTABLE_LVL3_IDX_PERTASK);
    let mut part = PageTablePart::new(address);
    part.map_2m(
        address,
        PhysAddr::new(0x400000),
        PTEntryFlags::task_data(),
        true,
    )
    .unwrap();
    let child = part.address().unwrap();
    let mut first = PageTable::allocate_new().unwrap();
    let mut second = PageTable::allocate_new().unwrap();
    assert!(first.populate_pgtbl_part(&part));
    assert!(!first.populate_pgtbl_part(&part));
    assert!(second.populate_pgtbl_part(&part));
    let entry = part.unmap_2m(address).unwrap();
    assert!(entry.present() && entry.huge());
    assert_eq!(entry.address(), PhysAddr::new(0x400000));
    assert!(first.phys_addr(address).is_err());
    drop(part);
    assert_eq!(
        Allocator::share_count(child.bits().into()).load(Ordering::Acquire),
        2
    );
    drop(first);
    drop(second);
    assert_reclaimed();
}

#[test]
fn inactive_percpu_replacement_flushes_before_releasing_old_branch() {
    let _memory = setup();
    let address = virt_from_idx(PGTABLE_LVL3_IDX_PERCPU);
    let mut old = PageTablePart::new(address);
    old.map_4k(address, PhysAddr::new(0x8000), PTEntryFlags::data(), false)
        .unwrap();
    let old_child = old.address().unwrap().bits();
    let mut next = PageTablePart::new(address);
    next.map_4k(address, PhysAddr::new(0xa000), PTEntryFlags::data(), false)
        .unwrap();
    let mut root = PageTable::allocate_new().unwrap();
    root.populate_pgtbl_part(&old);
    drop(old);
    let start = EVENTS.lock().len();
    // SAFETY: this test never activates any root in hardware.
    assert!(unsafe { root.populate_pgtbl_part_inactive(&next) });
    assert_eq!(root.phys_addr(address).unwrap(), PhysAddr::new(0xa000));
    let events = EVENTS.lock();
    let flush = events[start..]
        .iter()
        .position(|event| matches!(event, Event::Flush(..)))
        .unwrap();
    let free = events[start..]
        .iter()
        .position(|event| matches!(event, Event::Free(address) if *address == old_child))
        .unwrap();
    assert!(flush < free);
    drop(events);
    drop(next);
    drop(root);
    assert_reclaimed();
}

#[test]
fn failed_intermediate_allocation_rolls_back_staging_not_existing_mappings() {
    let _memory = setup();
    let mut table = PageTable::allocate_new().unwrap();
    table
        .map_4k(
            VirtAddr::new(0x4000),
            PhysAddr::new(0x8000),
            PTEntryFlags::data(),
            false,
        )
        .unwrap();
    let before = EVENTS.lock().len();
    FAIL_AFTER.store(1, Ordering::Relaxed);
    // A different 1 GiB region needs two intermediate tables.
    assert!(
        table
            .map_4k(
                VirtAddr::new(0x40000000),
                PhysAddr::new(0xa000),
                PTEntryFlags::data(),
                false
            )
            .is_err()
    );
    FAIL_AFTER.store(-1, Ordering::Relaxed);
    assert!(table.phys_addr(VirtAddr::new(0x40000000)).is_err());
    assert_eq!(
        table.phys_addr(VirtAddr::new(0x4000)).unwrap(),
        PhysAddr::new(0x8000)
    );
    let events = EVENTS.lock();
    assert_eq!(
        events[before..]
            .iter()
            .filter(|e| matches!(e, Event::Allocate(_)))
            .count(),
        1
    );
    assert_eq!(
        events[before..]
            .iter()
            .filter(|e| matches!(e, Event::Free(_)))
            .count(),
        1
    );
    drop(events);
    drop(table);
    assert_reclaimed();
}

#[test]
fn range_failure_retains_inserted_prefix_and_allows_caller_rollback() {
    let _memory = setup();
    let mut table = PageTable::allocate_new().unwrap();
    let start = VirtAddr::new(PAGE_SIZE_2M - PAGE_SIZE);
    table.branch_for_mutation(start).unwrap();
    FAIL_AFTER.store(2, Ordering::Relaxed);
    let region = MemoryRegion::new(start, 3 * PAGE_SIZE);
    assert!(
        table
            .map_region_4k(region, PhysAddr::new(0x8000), PTEntryFlags::data(), false)
            .is_err()
    );
    FAIL_AFTER.store(-1, Ordering::Relaxed);
    assert_eq!(table.phys_addr(start).unwrap(), PhysAddr::new(0x8000));
    assert!(table.phys_addr(start + PAGE_SIZE).is_err());
    assert!(table.phys_addr(start + 2 * PAGE_SIZE).is_err());
    table.unmap_region_4k(region);
    assert!(table.phys_addr(start).is_err());
    drop(table);
    assert_reclaimed();
}

#[test]
fn read_only_transition_preserves_data_and_history_and_rejects_non_global() {
    let _memory = setup();
    let mut table = PageTable::allocate_new().unwrap();
    let address = virt_from_idx(511) + 0x4000;
    table
        .map_4k(address, PhysAddr::new(0x8000), PTEntryFlags::data(), true)
        .unwrap();
    // SAFETY: no hardware or data users exist in this host test.
    unsafe { table.make_region_ro_4k(MemoryRegion::new(address, PAGE_SIZE)) }.unwrap();
    let entry = observed(&table, address);
    assert!(!entry.writable() && entry.is_shared());
    assert!(
        entry
            .flags()
            .contains(Flags::DIRTY | Flags::ACCESSED | Flags::NX)
    );
    table
        .map_4k(
            address + PAGE_SIZE,
            PhysAddr::new(0xa000),
            PTEntryFlags::task_data(),
            false,
        )
        .unwrap();
    // SAFETY: no hardware or data users exist in this host test.
    let result =
        unsafe { table.make_region_ro_4k(MemoryRegion::new(address + PAGE_SIZE, PAGE_SIZE)) };
    assert!(result.is_err());
    drop(table);
    assert_reclaimed();
}

#[test]
fn allocator_failure_and_invalid_alignment_are_reported_without_live_tree_damage() {
    let _memory = setup();
    FAIL_AFTER.store(0, Ordering::Relaxed);
    assert!(PageTable::allocate_new().is_err());
    FAIL_AFTER.store(-1, Ordering::Relaxed);
    let mut table = PageTable::allocate_new().unwrap();
    assert!(
        table
            .map_4k(
                VirtAddr::new(0x4001),
                PhysAddr::new(0x8000),
                PTEntryFlags::data(),
                false
            )
            .is_err()
    );
    assert!(
        table
            .map_4k(
                VirtAddr::new(0x4000),
                PhysAddr::new(0x8001),
                PTEntryFlags::data(),
                false
            )
            .is_err()
    );
    assert!(table.phys_addr(VirtAddr::new(0x4000)).is_err());
    assert!(
        table
            .map_4k(
                VirtAddr::new(0x4000),
                PhysAddr::new(0x8000),
                PTEntryFlags::empty(),
                false
            )
            .is_err()
    );
    table
        .map_4k(
            VirtAddr::new(0x4000),
            PhysAddr::new(0x8000),
            PTEntryFlags::data(),
            false,
        )
        .unwrap();
    assert!(
        table
            .map_4k(
                VirtAddr::new(0x4000),
                PhysAddr::new(0xa000),
                PTEntryFlags::data(),
                false
            )
            .is_err()
    );
    assert_eq!(
        table.phys_addr(VirtAddr::new(0x4000)).unwrap(),
        PhysAddr::new(0x8000)
    );
    drop(table);
    assert_reclaimed();
}

#[test]
fn owner_reference_allocation_failure_reclaims_the_unpublished_branch() {
    let _memory = setup();
    let mut table = Owner::new_empty().unwrap();
    let child = Allocator::allocate_zeroed_table_page().unwrap();
    // SAFETY: fresh zeroed, exclusively owned and never-published level-2 page.
    let child = unsafe { OwnedSubtree::<Arch, Allocator, Lvl<2>>::from_owned_root(child) };
    assert!(table.populate_owned(0, child).is_ok());
    let branch = Branch {
        index: 0,
        table,
        pending: PendingLock::new(MayNeedFlush::none()),
    };
    let mut occupied = Vec::new();
    while let Ok(address) = allocate_pages(0) {
        occupied.push(address);
    }
    assert!(BranchRef::new(branch).is_err());
    assert_reclaimed();
    for address in occupied {
        free_page(address);
    }
}

#[test]
fn flush_hooks_preserve_global_and_processor_scope() {
    let _memory = setup();
    let token = || MayNeedFlush::<paging::arch::x86_64::X86TlbFlushTok<Platform>>::all();
    token().flush_tlb_global_sync();
    token().flush_tlb_global_percpu();
    token().flush_tlb_ignore_global_sync();
    token().flush_tlb_percpu();
    let events = EVENTS.lock();
    assert!(matches!(
        events[0],
        Event::Flush(FlushScope::All, true, true)
    ));
    assert!(matches!(
        events[1],
        Event::Flush(FlushScope::All, true, false)
    ));
    assert!(matches!(
        events[2],
        Event::Flush(FlushScope::All, false, true)
    ));
    assert!(matches!(
        events[3],
        Event::Flush(FlushScope::All, false, false)
    ));
}

#[test]
fn ordinary_unmap_retains_obligation_without_flushing_under_outer_caller_locks() {
    let _memory = setup();
    let mut table = PageTable::allocate_new().unwrap();
    let address = virt_from_idx(511) + PAGE_SIZE;
    table
        .map_4k(address, PhysAddr::new(0x8000), PTEntryFlags::data(), false)
        .unwrap();
    let start = EVENTS.lock().len();
    table.unmap_4k(address);
    assert!(table.branch(511).unwrap().pending.lock().is_pending());
    assert!(table.phys_addr(address).is_err());
    assert!(
        !EVENTS.lock()[start..]
            .iter()
            .any(|event| matches!(event, Event::Flush(..)))
    );
    // Model the native caller's flush after releasing its outer guards.
    flush(
        FlushScope::Range {
            start: address.bits().into(),
            end: (address + PAGE_SIZE).bits().into(),
            level: PageLevel::Level0,
        },
        true,
        true,
    );
    drop(table);
    let events = EVENTS.lock();
    let first_flush = events[start..]
        .iter()
        .position(|event| matches!(event, Event::Flush(..)))
        .unwrap();
    let first_free = events[start..]
        .iter()
        .position(|event| matches!(event, Event::Free(_)))
        .unwrap();
    assert!(first_flush < first_free);
    drop(events);
    assert_reclaimed();
}

#[test]
fn boot_rebuild_skips_recursive_slot_and_preserves_4k_2m_1g_leaves() {
    let _memory = setup();
    let mut boot = PageBox::try_new([0usize; ENTRY_COUNT]).unwrap();
    let mut l2 = PageBox::try_new([0usize; ENTRY_COUNT]).unwrap();
    let mut l1 = PageBox::try_new([0usize; ENTRY_COUNT]).unwrap();
    let mut l0 = PageBox::try_new([0usize; ENTRY_COUNT]).unwrap();
    let parent = private_pte_mask() | 0x63;
    boot[PGTABLE_LVL3_IDX_PTE_SELFMAP] = boot.vaddr().bits() | parent;
    boot[511] = l2.vaddr().bits() | parent;
    l2[0] = l1.vaddr().bits() | parent;
    l1[0] = 0x400000 | private_pte_mask() | PTEntryFlags::data().bits() as usize | 0x80;
    l1[1] = l0.vaddr().bits() | parent;
    l0[0] = 0xa000 | native::shared_pte_mask() | PTEntryFlags::data_ro().bits() as usize;
    l2[2] = 0x80000000 | private_pte_mask() | PTEntryFlags::data().bits() as usize | 0x80;
    // SAFETY: these pinned initialized test tables are never loaded in hardware;
    // all table pointers resolve through the host's fixed identity mapping.
    let rebuilt = unsafe { PageTable::rebuild_boot_root(boot.vaddr()) }.unwrap();
    assert!(
        rebuilt
            .root
            .next_table_pa(PGTABLE_LVL3_IDX_PTE_SELFMAP)
            .is_none()
    );
    let address = virt_from_idx(511);
    for (offset, level, physical) in [
        (0, PageLevel::Level1, 0x400000),
        (PAGE_SIZE_2M, PageLevel::Level0, 0xa000),
        (2 * (1usize << 30), PageLevel::Level2, 0x80000000),
    ] {
        let target = address + offset;
        assert_eq!(rebuilt.root.walk(target.bits().into()).level(), level);
        assert_eq!(
            rebuilt.phys_addr(target + 17).unwrap(),
            PhysAddr::new(physical + 17)
        );
        assert!(
            !observed(&rebuilt, target).user(),
            "ancestor USER restriction must survive"
        );
    }

    assert!(observed(&rebuilt, address + PAGE_SIZE_2M).is_shared());
    // SAFETY: rebuilt's root and descendants remain pinned for this raw read.
    let frame = unsafe {
        frame_from_root(
            phys_to_virt(rebuilt.cr3_value()),
            address + PAGE_SIZE_2M + 17,
        )
    }
    .unwrap();
    assert_eq!(frame.address(), PhysAddr::new(0xa011));
    assert_ne!(frame.page_frame().bits() & native::shared_pte_mask(), 0);
    drop(rebuilt);
    drop((boot, l2, l1, l0));
    assert_reclaimed();
}

#[test]
fn boot_rebuild_rejects_unrepresentable_cache_attributes_without_leaking() {
    let _memory = setup();
    let mut boot = PageBox::try_new([0usize; ENTRY_COUNT]).unwrap();
    let mut l2 = PageBox::try_new([0usize; ENTRY_COUNT]).unwrap();
    let parent = private_pte_mask() | 0x63;
    boot[511] = l2.vaddr().bits() | parent;
    // The existing boot-image generator never uses huge-page PAT.
    l2[0] = 0x80000000 | private_pte_mask() | PTEntryFlags::data().bits() as usize | 0x1080;
    // SAFETY: pinned, initialized, never-active host test boot tables.
    let result = unsafe { PageTable::rebuild_boot_root(boot.vaddr()) };
    assert!(result.is_err());
    drop((boot, l2));
    assert_reclaimed();
}
