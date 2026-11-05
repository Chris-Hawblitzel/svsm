# Opt-in VeriSMo paging backend

The `svsm` package's **`verismo-paging`** feature selects the adjacent
`verios-pagetable-beta` package (library name `paging`). Native paging remains
the default. This is a local integration: the dependency expects this checkout
at `source/paging/compat/svsm`.

## Building and testing

Use the ordinary SVSM image recipes, adding the kernel feature:

```shell
make FEATURES=verismo-paging
# Equivalently, xbuild accepts the explicit package-qualified spelling:
cargo xbuild -f svsm:verismo-paging ./configs/qemu-target.json
```

For a linked kernel without image assembly:

```shell
export CARGO_TARGET_DIR=target/verismo-integration
cargo build -p svsm --bin svsm --target x86_64-unknown-none \
    --release --features verismo-paging --locked
```

Executable host tests use SVSM's actual page allocator and selected backend,
with architectural flush hooks replaced by test event recording:

```shell
export CARGO_TARGET_DIR=target/verismo-integration
cargo test -p svsm --lib --target x86_64-unknown-linux-gnu \
    --features verismo-paging --locked
# Focused adapter tests:
cargo test -p svsm --lib --target x86_64-unknown-linux-gnu \
    --features verismo-paging --locked mm::pagetable::tests
```

Use the checkout's pinned Rust toolchain. A working host linker is necessary
even for bare-metal builds, because build scripts and proc macros run on the
host. Initialize the recorded `packit` submodule first if it is absent.

The target configurations specify `target_min_page="4kib"` for paging.
SVSM's optional Verus dependencies are aligned with the adjacent workspace's
exact prereleases because Cargo cannot resolve their previous, different
exact prereleases in the same `0.0.0` compatibility range. Checking the
`verus`/`verus_all` Cargo features is not a formal proof run.

## Architecture and ownership

The implementation is `kernel/src/mm/pagetable_verismo.rs`; the existing
`mm::pagetable` module path selects it only with the feature enabled.
Native entry encodings, platform initialization, guest page-table walking,
and early boot utilities remain available.

SVSM uses four-level, 512-entry, 4 KiB x86-64 tables. There is no single
Linux-style kernel/user ownership split:

| Root entry | Purpose |
|---|---|
| 0–255 | Lazily populated userspace VMRs |
| 493 | Native recursive mapping; absent from rebuilt roots |
| 508 | Per-task kernel VMR |
| 510 | Per-CPU VMR; replaced when a task migrates |
| 511 | Shared kernel, heap/direct-map, and global mappings |

Each VMR part retains a counted reference to a branch owner. A branch owner
uses a fresh level-3 controller with exactly one populated root slot and checks every
mutation against that slot. Its controller owns its tables. Hardware roots
use a no-owned-entries policy and borrow those branches through the paging
crate's explicit sharing API. Every borrowing root retains the corresponding
owner reference.

Operations through a borrowed kernel/per-CPU/task root are routed to its
retained branch owner. No ownership-unchecked mutation or foreign populated
root adoption is used. Roots release their paging borrow counts before
dropping owner references. Tree destruction reclaims table allocations, not
the mapped data frames managed by SVSM VMMs.

Owner references are backed by a fallible `PageBox` allocation rather than
`Arc::new`, which cannot report allocation failure on the pinned stable
toolchain. Cloning a reference does not allocate. The last reference drops
the branch controller and reclaims its owner allocation.

This design costs an extra owner root and owner-metadata page per branch.
Each table allocation also uses two contiguous pages: the hardware table and a metadata page containing
its per-frame share counter. The metadata is not exposed
to hardware. Both pages are reclaimed together. This avoids an allocating
global registry and makes allocator failures explicit; it is not a memory
or performance optimization.

## Bootstrap and migration

The boot image still creates the native recursive hierarchy. Early heap
expansion and translation use that hierarchy unchanged.

After heap/fixed-mapping initialization, `PageTable::rebuild_boot_root` walks
the trusted boot tree and builds new acyclic roots using `new_empty`. It
copies 4 KiB, 2 MiB, and 1 GiB leaves, preserves the flags used by the current
boot image, and collapses ancestor write/user/execute restrictions into the
new leaves. Slot 493 is deliberately excluded. Unsupported cache attributes
(including PAT) cause an error rather than being silently lost; the existing
boot-image generator does not generate them.

The original boot table pages remain reserved. They are not handed to the
paging allocator or reclaimed while the original root is active. The BSP
loads the rebuilt root only after per-CPU setup. Thereafter allocator
translation walks the current CR3 atomically through the fixed heap mapping,
instead of reading recursive addresses. APs and tasks use the rebuilt format.

Ordinary VMR population installs absent slots or recognizes the same owner.
Replacing a different owner requires the separate unsafe inactive-root
population path. The scheduler supplies its existing handshake: switch away
from the outgoing task's CR3, clear its active flag, then exclusively claim
the incoming task before updating its per-CPU slot and loading its CR3.
The replacement path completes distributed invalidation before releasing
the old borrow. Active-root replacement is not supported.

## Mutation, confidentiality, and TLB maintenance

Private/shared address masks and allowed flags come from SVSM's platform
initialization, not hard-coded SEV bit positions. The adapter supports both
a set private C-bit and a set shared GPA tag. Physical translations strip
both; page-frame queries retain the shared tag. Table pointers are private.

Runtime mappings retain SVSM's 4 KiB/2 MiB interfaces. Visibility changes
split huge leaves through paging's serialized split implementation and
preserve neighboring mappings and hardware A/D history. Returned read-only,
unmap, and visibility flush obligations are conservatively retained in branch
metadata. Existing SVSM callers perform their synchronous flushes after
releasing the relevant page-table/VMR guards, before platform/cache transitions
or data/address reuse. Adding a shootdown while those outer locks are held
could deadlock an IRQ-disabled waiter, so the adapter does not do that.
Residual conservative tokens are discarded only when the last owner reference
is destroyed: every borrowing root has then been excluded from hardware and
flushed before releasing its owner reference. Structural huge-page splitting
still performs the paging crate's required synchronous maintenance.

Range insertion retains successfully mapped prefixes on error. VMR insertion
keeps its existing unmap/rollback path. Failed missing-branch construction is
privately rolled back by paging rather than retaining native empty
intermediate tables; no existing leaf is removed by that failure.

The paging crate rejects overlapping insertion rather than overwriting a
present leaf. SVSM's existing VMR/global-range allocation paths insert into
disjoint regions; permissions and confidentiality use their dedicated update
interfaces. A caller needing native-style replacement of an existing
physical leaf needs an explicit atomic replacement API in paging; an
unmap/map gap or raw PTE overwrite is not a substitute.

All flush hooks preserve SVSM's global/non-global and local/distributed
dispatch. A 1 GiB/structural scope widens to a full flush because SVSM's range
interface exposes only 4 KiB/2 MiB granularity. Distributed calls retain
platform dispatch, including SNP-specific mechanisms and synchronous IPI
acknowledgments; no local-only fallback is introduced after SMP startup.

The backend uses paging's built-in global ticket mutex by default.
`paging/striped-locks` selects static physical-address stripes and
`paging/spin-mutex` selects the non-FIFO spin mutex; no custom lock provider is
installed. Outer operation guards mask preemption with `TPR_LOCK` while
potentially locking paging operations and pending-obligation updates run.
Some callers already have IRQs disabled, so waiting only for interrupt
delivery can deadlock against a writer holding a content lock during a
distributed flush. Contended waits and IPI send waits therefore poll
**only explicitly marked lock-independent translation shootdowns**.
Other IPI messages remain queued. This preserves the native IPI completion
accounting without enabling unrelated interrupts or re-entering paging.
The required `paging/lock-wait-hook` feature calls the exported
`paging_lock_wait()` from the mutex's `RelaxStrategy`, without relinquishing
queued ticket positions.

An explicit readiness flag gates both the wait hook and operation guards.
Single-threaded, IRQ-disabled BSP bootstrap cannot access per-CPU keys yet,
so neither path touches GS before readiness is published immediately after
BSP `setup_on_cpu()`. AP entry installs its own per-CPU keys before paging.
The fatal NMI path does not enter paging; paging re-entry from NMI remains
unsupported. The host hook is a no-op and host guards access no privileged
or per-CPU state. These context rules are required, not inferred from a mutex.

Dropping a root requires hardware exclusion. The unsafe load contract and
the scheduler's lifecycle provide that exclusion; destruction checks that
the root is not the current CPU's CR3 and flushes before reclaiming pages.
Reference counts alone are not hardware-quiescence evidence.

## Validation and limitations

Validation in this environment:

* Native host suite: **141 passed, 0 failed, 36 ignored**.
* Opt-in host suite: **159 passed, 0 failed, 36 ignored**, including seventeen
  adapter tests and an executable IPI polling/classification test. The complete
  opt-in suite passed with each of the four global/striped × ticket/spin mutex
  configurations, including a host-hook/guard context test.
* Native and opt-in release kernel binaries linked successfully for
  `x86_64-unknown-none`; the opt-in striped/spin combination also linked.
* Strict opt-in Clippy passed for host library/tests and bare-metal targets.
* `cargo fmt -p svsm --check` passed.
* Both `verus` and `verus_all` Cargo configurations checked with the backend.

Adapter tests exercise real mapping/translation, flags and confidentiality
bits, huge splitting and failure preservation, concurrent borrowers, prefix
rollback, unmap return values, root/VMR ownership and exact reclamation,
recursive-root rebuilding, unsupported-attribute rejection, and flush scopes
and ordering. Ignored existing tests are not counted as evidence.

No SVSM hardware boot or QEMU boot was executed here. The tests do not prove
hardware page-walk quiescence, SMP interrupt delivery, SEV-SNP/TDX transitions,
or performance. The kernel/image must still be exercised on the intended
platform before deployment. No new formal verification claim is made for
this adapter or its IPI integration.
