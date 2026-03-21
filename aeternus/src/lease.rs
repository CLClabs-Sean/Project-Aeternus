//! # KV-Cache Lease System
//!
//! Manages a host-side memory pool (targeting 96GB DDR) for latent block
//! orchestration between agents. Zero-copy hand-off via shared references.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Unique identifier for a memory lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeaseId(u64);

/// A reference-counted handle to a region of the host memory pool.
/// Agents share leases via `clone()` — no memcpy, just a refcount bump.
#[derive(Debug)]
pub struct Lease {
    pub id: LeaseId,
    offset: usize,
    size: usize,
    refcount: Arc<AtomicU32>,
    pool_ptr: *mut u8,
}

impl Lease {
    /// Zero-copy pointer to the leased region.
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.pool_ptr.add(self.offset) }
    }

    /// Mutable pointer (for writing reconstructed latents).
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.pool_ptr.add(self.offset) }
    }

    /// View the leased region as a typed slice.
    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        let count = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const T, count) }
    }

    /// View the leased region as a mutable typed slice.
    pub fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        let count = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut T, count) }
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Current reference count.
    pub fn refcount(&self) -> u32 {
        self.refcount.load(Ordering::Relaxed)
    }

    /// Share this lease with another agent (zero-copy, refcount bump).
    pub fn share(&self) -> Lease {
        self.refcount.fetch_add(1, Ordering::Relaxed);
        Lease {
            id: self.id,
            offset: self.offset,
            size: self.size,
            refcount: Arc::clone(&self.refcount),
            pool_ptr: self.pool_ptr,
        }
    }
}

// Safety: The pool backing memory is stable (boxed Vec), and access is
// coordinated by the LeasePool's allocation tracking.
unsafe impl Send for Lease {}
unsafe impl Sync for Lease {}

/// Allocation slot tracked by the pool.
struct LeaseSlot {
    offset: usize,
    size: usize,
    refcount: Arc<AtomicU32>,
}

/// Host memory pool with first-fit allocation and coalescing.
pub struct LeasePool {
    pool: Box<[u8]>,
    capacity: usize,
    next_id: u64,
    allocations: BTreeMap<LeaseId, LeaseSlot>,
    /// Free list: sorted by offset, coalesced on release.
    free_list: Vec<(usize, usize)>, // (offset, size)
}

impl LeasePool {
    /// Create a new pool with the given capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        let pool = vec![0u8; capacity].into_boxed_slice();
        Self {
            pool,
            capacity,
            next_id: 0,
            allocations: BTreeMap::new(),
            free_list: vec![(0, capacity)],
        }
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total free space.
    pub fn free_space(&self) -> usize {
        self.free_list.iter().map(|(_, s)| *s).sum()
    }

    /// Number of active leases.
    pub fn active_leases(&self) -> usize {
        self.allocations.len()
    }

    /// Acquire a lease of `size` bytes. Returns None if OOM.
    pub fn acquire(&mut self, size: usize) -> Option<Lease> {
        // Align to 64 bytes for GPU DMA compatibility.
        let aligned_size = (size + 63) & !63;

        // First-fit search.
        let slot_idx = self.free_list.iter()
            .position(|(_, s)| *s >= aligned_size)?;

        let (offset, free_size) = self.free_list[slot_idx];

        // Shrink or remove the free block.
        if free_size == aligned_size {
            self.free_list.remove(slot_idx);
        } else {
            self.free_list[slot_idx] = (offset + aligned_size, free_size - aligned_size);
        }

        let id = LeaseId(self.next_id);
        self.next_id += 1;

        let refcount = Arc::new(AtomicU32::new(1));

        self.allocations.insert(id, LeaseSlot {
            offset,
            size: aligned_size,
            refcount: Arc::clone(&refcount),
        });

        Some(Lease {
            id,
            offset,
            size: aligned_size,
            refcount,
            pool_ptr: self.pool.as_mut_ptr(),
        })
    }

    /// Release a lease. The backing memory is freed when all shared
    /// references have been released.
    pub fn release(&mut self, lease: Lease) -> bool {
        let prev = lease.refcount.fetch_sub(1, Ordering::Relaxed);
        if prev == 1 {
            // Last reference — free the slot.
            if let Some(slot) = self.allocations.remove(&lease.id) {
                self.return_to_free_list(slot.offset, slot.size);
                return true;
            }
        }
        false
    }

    /// Return a block to the free list with coalescing.
    fn return_to_free_list(&mut self, offset: usize, size: usize) {
        let end = offset + size;

        // Find insertion point (sorted by offset).
        let insert_pos = self.free_list.iter()
            .position(|(o, _)| *o > offset)
            .unwrap_or(self.free_list.len());

        // Try to coalesce with the previous block.
        let mut merged_offset = offset;
        let mut merged_size = size;
        let mut remove_prev = false;

        if insert_pos > 0 {
            let (prev_off, prev_size) = self.free_list[insert_pos - 1];
            if prev_off + prev_size == offset {
                merged_offset = prev_off;
                merged_size += prev_size;
                remove_prev = true;
            }
        }

        // Try to coalesce with the next block.
        let mut remove_next = false;
        if insert_pos < self.free_list.len() {
            let (next_off, next_size) = self.free_list[insert_pos];
            if merged_offset + merged_size == next_off {
                merged_size += next_size;
                remove_next = true;
            }
        }

        // Apply coalescing.
        if remove_prev && remove_next {
            self.free_list[insert_pos - 1] = (merged_offset, merged_size);
            self.free_list.remove(insert_pos);
        } else if remove_prev {
            self.free_list[insert_pos - 1] = (merged_offset, merged_size);
        } else if remove_next {
            self.free_list[insert_pos] = (merged_offset, merged_size);
        } else {
            self.free_list.insert(insert_pos, (merged_offset, merged_size));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_lifecycle() {
        let mut pool = LeasePool::new(4096);
        assert_eq!(pool.free_space(), 4096);

        let lease = pool.acquire(100).expect("should allocate");
        assert!(lease.size() >= 100);
        assert_eq!(lease.refcount(), 1);
        assert_eq!(pool.active_leases(), 1);

        pool.release(lease);
        assert_eq!(pool.active_leases(), 0);
        assert_eq!(pool.free_space(), 4096);
    }

    #[test]
    fn zero_copy_sharing() {
        let mut pool = LeasePool::new(4096);
        let lease_a = pool.acquire(256).expect("alloc");

        // Share with another "agent" — no memcpy.
        let lease_b = lease_a.share();
        assert_eq!(lease_a.refcount(), 2);
        assert_eq!(lease_b.refcount(), 2);
        assert_eq!(lease_a.as_ptr(), lease_b.as_ptr());

        // Release one — memory should NOT be freed yet.
        let freed = pool.release(lease_a);
        assert!(!freed);
        assert_eq!(pool.active_leases(), 1);

        // Release the other — NOW it's freed.
        let freed = pool.release(lease_b);
        assert!(freed);
        assert_eq!(pool.active_leases(), 0);
    }

    #[test]
    fn coalescing() {
        let mut pool = LeasePool::new(4096);
        let a = pool.acquire(128).unwrap();
        let b = pool.acquire(128).unwrap();
        let c = pool.acquire(128).unwrap();

        // Release middle block.
        pool.release(b);
        assert_eq!(pool.free_list.len(), 2); // gap + tail

        // Release first block — should coalesce with gap.
        pool.release(a);

        // Release third — should coalesce everything.
        pool.release(c);
        assert_eq!(pool.free_list.len(), 1);
        assert_eq!(pool.free_space(), 4096);
    }

    #[test]
    fn typed_slice_access() {
        let mut pool = LeasePool::new(4096);
        let mut lease = pool.acquire(256).unwrap();

        // Write f32 data via typed slice.
        let data: &mut [f32] = lease.as_mut_slice();
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f32;
        }

        // Read back.
        let read: &[f32] = lease.as_slice();
        assert_eq!(read[0], 0.0);
        assert_eq!(read[1], 1.0);

        pool.release(lease);
    }

    #[test]
    fn oom_returns_none() {
        let mut pool = LeasePool::new(256);
        let _a = pool.acquire(256).unwrap();
        assert!(pool.acquire(64).is_none());
    }

    #[test]
    fn alignment_64() {
        let mut pool = LeasePool::new(4096);
        let a = pool.acquire(1).unwrap(); // request 1 byte
        assert_eq!(a.size(), 64); // aligned to 64
        let b = pool.acquire(1).unwrap();
        assert_eq!(b.size(), 64);
        // Pointers should be 64-byte aligned apart.
        let diff = (b.as_ptr() as usize) - (a.as_ptr() as usize);
        assert_eq!(diff, 64);
        pool.release(a);
        pool.release(b);
    }
}
