/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  amd_detail/hip_cooperative_groups.h
 *
 *  @brief Device side implementation of `Cooperative Group` feature.
 *
 *  Defines new types and device API wrappers related to `Cooperative Group`
 *  feature, which the programmer can directly use in his kernel(s) in order to
 *  make use of this feature.
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H

#if __cplusplus
#include <hip/amd_detail/hip_cooperative_groups_helper.h>

namespace cooperative_groups {

/** \brief The base type of all cooperative group types
 *
 *  \details Holds the key properties of a constructed cooperative group types
 *           object, like the group type, its size, etc
 */
class thread_group {
 protected:
  uint32_t _type;  // thread_group type
  uint32_t _size;  // total number of threads in the tread_group
  uint64_t _mask;  // Lanemask for coalesced and tiled partitioned group types,
                   // LSB represents lane 0, and MSB represents lane 63

  // Construct a thread group, and set thread group type and other essential
  // thread group properties. This generic thread group is directly constructed
  // only when the group is supposed to contain only the calling the thread
  // (throurh the API - `this_thread()`), and in all other cases, this thread
  // group object is a sub-object of some other derived thread group object
  __CG_QUALIFIER__ thread_group(internal::group_type type, uint32_t size,
                                uint64_t mask = (uint64_t)0) {
    _type = type;
    _size = size;
    _mask = mask;
  }

  struct _tiled_info {
    bool is_tiled;
    unsigned int size;
  } tiled_info;

  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend class thread_block;

 public:
  // Total number of threads in the thread group, and this serves the purpose
  // for all derived cooperative group types since their `size` is directly
  // saved during the construction
  __CG_QUALIFIER__ uint32_t size() const { return _size; }
  __CG_QUALIFIER__ unsigned int cg_type() const { return _type; }
  // Rank of the calling thread within [0, size())
  __CG_QUALIFIER__ uint32_t thread_rank() const;
  // Is this cooperative group type valid?
  __CG_QUALIFIER__ bool is_valid() const;
  // synchronize the threads in the thread group
  __CG_QUALIFIER__ void sync() const;
};

/** \brief The multi-grid cooperative group type
 *
 *  \details Represents an inter-device cooperative group type where the
 *           participating threads within the group spans across multple
 *           devices, running the (same) kernel on these devices
 */
class multi_grid_group : public thread_group {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  friend __CG_QUALIFIER__ multi_grid_group this_multi_grid();

 protected:
  // Construct mutli-grid thread group (through the API this_multi_grid())
  explicit __CG_QUALIFIER__ multi_grid_group(uint32_t size)
      : thread_group(internal::cg_multi_grid, size) {}

 public:
  // Number of invocations participating in this multi-grid group. In other
  // words, the number of GPUs
  __CG_QUALIFIER__ uint32_t num_grids() { return internal::multi_grid::num_grids(); }
  // Rank of this invocation. In other words, an ID number within the range
  // [0, num_grids()) of the GPU, this kernel is running on
  __CG_QUALIFIER__ uint32_t grid_rank() { return internal::multi_grid::grid_rank(); }
  __CG_QUALIFIER__ uint32_t thread_rank() const { return internal::multi_grid::thread_rank(); }
  __CG_QUALIFIER__ bool is_valid() const { return internal::multi_grid::is_valid(); }
  __CG_QUALIFIER__ void sync() const { internal::multi_grid::sync(); }
};

/** \brief User exposed API interface to construct multi-grid cooperative
 *         group type object - `multi_grid_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `multi_grid_group`. Instead, he should construct it through this
 *           API function
 */
__CG_QUALIFIER__ multi_grid_group this_multi_grid() {
  return multi_grid_group(internal::multi_grid::size());
}

/** \brief The grid cooperative group type
 *
 *  \details Represents an inter-workgroup cooperative group type where the
 *           participating threads within the group spans across multiple
 *           workgroups running the (same) kernel on the same device
 */
class grid_group : public thread_group {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  friend __CG_QUALIFIER__ grid_group this_grid();

 protected:
  // Construct grid thread group (through the API this_grid())
  explicit __CG_QUALIFIER__ grid_group(uint32_t size) : thread_group(internal::cg_grid, size) {}

 public:
  __CG_QUALIFIER__ uint32_t thread_rank() const { return internal::grid::thread_rank(); }
  __CG_QUALIFIER__ bool is_valid() const { return internal::grid::is_valid(); }
  __CG_QUALIFIER__ void sync() const { internal::grid::sync(); }
};

/** \brief User exposed API interface to construct grid cooperative group type
 *         object - `grid_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `multi_grid_group`. Instead, he should construct it through this
 *           API function
 */
__CG_QUALIFIER__ grid_group this_grid() { return grid_group(internal::grid::size()); }

/** \brief   The workgroup (thread-block in CUDA terminology) cooperative group
 *           type
 *
 *  \details Represents an intra-workgroup cooperative group type where the
 *           participating threads within the group are exactly the same threads
 *           which are participated in the currently executing `workgroup`
 */
class thread_block : public thread_group {
  // Only these friend functions are allowed to construct an object of thi
  // class and access its resources
  friend __CG_QUALIFIER__ thread_block this_thread_block();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_block& parent,
                                                       unsigned int tile_size);

 protected:
  // Construct a workgroup thread group (through the API this_thread_block())
  explicit __CG_QUALIFIER__ thread_block(uint32_t size)
      : thread_group(internal::cg_workgroup, size) {}

  __CG_QUALIFIER__ thread_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);
    // Invalid tile size, assert
    if (!tile_size || (tile_size > WAVEFRONT_SIZE) || !pow2) {
      assert(false && "invalid tile size");
    }

    thread_group tiledGroup = thread_group(internal::cg_tiled_group, tile_size);
    tiledGroup.tiled_info.size = tile_size;
    tiledGroup.tiled_info.is_tiled = true;
    return tiledGroup;
  }

 public:
  // 3-dimensional block index within the grid
  __CG_QUALIFIER__ dim3 group_index() { return internal::workgroup::group_index(); }
  // 3-dimensional thread index within the block
  __CG_QUALIFIER__ dim3 thread_index() { return internal::workgroup::thread_index(); }
  __CG_QUALIFIER__ uint32_t thread_rank() const { return internal::workgroup::thread_rank(); }
  __CG_QUALIFIER__ bool is_valid() const { return internal::workgroup::is_valid(); }
  __CG_QUALIFIER__ void sync() const { internal::workgroup::sync(); }
};

/** \brief   User exposed API interface to construct workgroup cooperative
 *           group type object - `thread_block`.
 *
 *  \details User is not allowed to directly construct an object of type
 *           `thread_block`. Instead, he should construct it through this API
 *           function.
 */
__CG_QUALIFIER__ thread_block this_thread_block() {
  return thread_block(internal::workgroup::size());
}

/** \brief   The tiled_group cooperative group type
 *
 *  \details Represents one tiled thread group in a wavefront.
 *           This group type also supports sub-wave level intrinsics.
 */

class tiled_group : public thread_group {
 private:
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ tiled_group tiled_partition(const tiled_group& parent,
                                                      unsigned int tile_size);

  __CG_QUALIFIER__ tiled_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);

    if (!tile_size || (tile_size > WAVEFRONT_SIZE) || !pow2) {
      assert(false && "invalid tile size");
    }

    if (size() <= tile_size) {
      return (*this);
    }

    tiled_group tiledGroup = tiled_group(tile_size);
    tiledGroup.tiled_info.is_tiled = true;
    return tiledGroup;
  }

 protected:
  explicit __CG_QUALIFIER__ tiled_group(unsigned int tileSize)
      : thread_group(internal::cg_tiled_group, tileSize) {
    tiled_info.size = tileSize;
    tiled_info.is_tiled = true;
  }

 public:
  __CG_QUALIFIER__ unsigned int size() const { return (tiled_info.size); }

  __CG_QUALIFIER__ unsigned int thread_rank() const {
    return (internal::workgroup::thread_rank() & (tiled_info.size - 1));
  }

  __CG_QUALIFIER__ void sync() const {
    // enforce memory ordering for memory instructions.
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
  }
};

/**
 *  Implemenation of all publicly exposed base class APIs
 */
__CG_QUALIFIER__ uint32_t thread_group::thread_rank() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      return (static_cast<const multi_grid_group*>(this)->thread_rank());
    }
    case internal::cg_grid: {
      return (static_cast<const grid_group*>(this)->thread_rank());
    }
    case internal::cg_workgroup: {
      return (static_cast<const thread_block*>(this)->thread_rank());
    }
    case internal::cg_tiled_group: {
      return (static_cast<const tiled_group*>(this)->thread_rank());
    }
    default: {
      assert(false && "invalid cooperative group type");
      return -1;
    }
  }
}

__CG_QUALIFIER__ bool thread_group::is_valid() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      return (static_cast<const multi_grid_group*>(this)->is_valid());
    }
    case internal::cg_grid: {
      return (static_cast<const grid_group*>(this)->is_valid());
    }
    case internal::cg_workgroup: {
      return (static_cast<const thread_block*>(this)->is_valid());
    }
    case internal::cg_tiled_group: {
      return (static_cast<const tiled_group*>(this)->is_valid());
    }
    default: {
      assert(false && "invalid cooperative group type");
      return false;
    }
  }
}

__CG_QUALIFIER__ void thread_group::sync() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      static_cast<const multi_grid_group*>(this)->sync();
      break;
    }
    case internal::cg_grid: {
      static_cast<const grid_group*>(this)->sync();
      break;
    }
    case internal::cg_workgroup: {
      static_cast<const thread_block*>(this)->sync();
      break;
    }
    case internal::cg_tiled_group: {
      static_cast<const tiled_group*>(this)->sync();
      break;
    }
    default: {
      assert(false && "invalid cooperative group type");
    }
  }
}

/**
 *  Implemenation of publicly exposed `wrapper` APIs on top of basic cooperative
 *  group type APIs
 */
template <class CGTy> __CG_QUALIFIER__ uint32_t group_size(CGTy const& g) { return g.size(); }

template <class CGTy> __CG_QUALIFIER__ uint32_t thread_rank(CGTy const& g) {
  return g.thread_rank();
}

template <class CGTy> __CG_QUALIFIER__ bool is_valid(CGTy const& g) { return g.is_valid(); }

template <class CGTy> __CG_QUALIFIER__ void sync(CGTy const& g) { g.sync(); }

template <unsigned int tileSize> class tile_base {
 protected:
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;

 public:
  // Rank of the thread within this tile
  _CG_STATIC_CONST_DECL_ unsigned int thread_rank() {
    return (internal::workgroup::thread_rank() & (numThreads - 1));
  }

  // Number of threads within this tile
  __CG_STATIC_QUALIFIER__ unsigned int size() { return numThreads; }
};

template <unsigned int size> class thread_block_tile_base : public tile_base<size> {
  static_assert(is_valid_tile_size<size>::value,
                "Tile size is either not a power of 2 or greater than the wavefront size");
  using tile_base<size>::numThreads;

 public:
  __CG_STATIC_QUALIFIER__ void sync() {
    // enforce ordering for memory instructions
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
  }

  template <class T> __CG_QUALIFIER__ T shfl(T var, int srcRank) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");
    return (__shfl(var, srcRank, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");
    return (__shfl_down(var, lane_delta, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");
    return (__shfl_up(var, lane_delta, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_xor(T var, unsigned int laneMask) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");
    return (__shfl_xor(var, laneMask, numThreads));
  }
};

/** \brief   Group type - thread_block_tile
 *
 *  \details  Represents one tile of thread group.
 */

template <unsigned int tileSize, class ParentCGTy = void>
class thread_block_tile_type : public thread_block_tile_base<tileSize>, public tiled_group {
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;

  friend class thread_block_tile_type<tileSize, ParentCGTy>;

  typedef thread_block_tile_base<numThreads> tbtBase;

 protected:
  __CG_QUALIFIER__ thread_block_tile_type() : tiled_group(numThreads) {
    tiled_info.size = numThreads;
    tiled_info.is_tiled = true;
  }

 public:
  using tbtBase::size;
  using tbtBase::sync;
  using tbtBase::thread_rank;
};


/** \brief   User exposed API to partition groups.
 *
 *  \details A collective operation that partitions the parent group into a one-dimensional,
 *           row-major, tiling of subgroups.
 */

__CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent, unsigned int tile_size) {
  if (parent.cg_type() == internal::cg_tiled_group) {
    const tiled_group* cg = static_cast<const tiled_group*>(&parent);
    return cg->new_tiled_group(tile_size);
  } else {
    const thread_block* tb = static_cast<const thread_block*>(&parent);
    return tb->new_tiled_group(tile_size);
  }
}

// Thread block type overload
__CG_QUALIFIER__ thread_group tiled_partition(const thread_block& parent, unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

// Coalesced group type overload
__CG_QUALIFIER__ tiled_group tiled_partition(const tiled_group& parent, unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

template <unsigned int size, class ParentCGTy> class thread_block_tile;

namespace impl {
template <unsigned int size, class ParentCGTy> class thread_block_tile_internal;

template <unsigned int size, class ParentCGTy>
class thread_block_tile_internal : public thread_block_tile_type<size, ParentCGTy> {
 protected:
  template <unsigned int tbtSize, class tbtParentT>
  __CG_QUALIFIER__ thread_block_tile_internal(
      const thread_block_tile_internal<tbtSize, tbtParentT>& g)
      : thread_block_tile_type<size, ParentCGTy>() {}

  __CG_QUALIFIER__ thread_block_tile_internal(const thread_block& g)
      : thread_block_tile_type<size, ParentCGTy>() {}
};
}  // namespace impl

template <unsigned int size, class ParentCGTy>
class thread_block_tile : public impl::thread_block_tile_internal<size, ParentCGTy> {
 protected:
  __CG_QUALIFIER__ thread_block_tile(const ParentCGTy& g)
      : impl::thread_block_tile_internal<size, ParentCGTy>(g) {}

 public:
  __CG_QUALIFIER__ operator thread_block_tile<size, void>() const {
    return thread_block_tile<size, void>(*this);
  }
};


template <unsigned int size>
class thread_block_tile<size, void> : public impl::thread_block_tile_internal<size, void> {
  template <unsigned int, class ParentCGTy> friend class thread_block_tile;

 protected:
 public:
  template <class ParentCGTy>
  __CG_QUALIFIER__ thread_block_tile(const thread_block_tile<size, ParentCGTy>& g)
      : impl::thread_block_tile_internal<size, void>(g) {}
};

template <unsigned int size, class ParentCGTy = void> class thread_block_tile;

namespace impl {
template <unsigned int size, class ParentCGTy = void> struct tiled_partition_internal;

template <unsigned int size>
struct tiled_partition_internal<size, thread_block> : public thread_block_tile<size, thread_block> {
  __CG_QUALIFIER__ tiled_partition_internal(const thread_block& g)
      : thread_block_tile<size, thread_block>(g) {}
};

}  // namespace impl

/** \brief   User exposed API to partition groups.
 *
 *  \details  This constructs a templated class derieved from thread_group.
 *            The template defines tile size of the new thread group at compile time.
 */
template <unsigned int size, class ParentCGTy>
__CG_QUALIFIER__ thread_block_tile<size, ParentCGTy> tiled_partition(const ParentCGTy& g) {
  static_assert(is_valid_tile_size<size>::value,
                "Tiled partition with size > wavefront size. Currently not supported ");
  return impl::tiled_partition_internal<size, ParentCGTy>(g);
}
}  // namespace cooperative_groups

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H
