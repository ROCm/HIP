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
 *  @file  hcc_detail/hip_cooperative_groups.h
 *
 *  @brief Device side implementation of `Cooperative Group` feature.
 *
 *  Defines new types and device API wrappers related to `Cooperative Group`
 *  feature, which the programmer can directly use in his kernel(s) in order to
 *  make use of this feature.
 *
 *  The `Cooperative Group` feature is supported only for hip/vdi runtime, as
 *  such, the the feature is implemented in case of hip/vdi runtime
 */
#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_H

#if __cplusplus && __HIP_VDI__
#include <hip/hcc_detail/hip_cooperative_groups_helper.h>

namespace cooperative_groups {

// Forward declaration of class which represents templated tiled cooperative
// groups
template <uint32_t tile_sz>
class thread_block_tile;

/** \brief The base type of all cooperative group types
 *
 *  \details Holds the key properties of a constructed cooperative group type
 *           object, like the group type, its size, etc
 */
class thread_group {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  friend __CG_QUALIFIER__ thread_group this_thread();
  friend __CG_QUALIFIER__ thread_group tiled_partition(
      const thread_group& parent, uint32_t tile_sz);
  template <uint32_t tile_sz>
  friend __CG_QUALIFIER__ thread_block_tile<tile_sz> tiled_partition(
      const thread_group& parent);

 protected:
  uint32_t _type; // thread_group type
  uint32_t _size; // total number of threads in the tread_group
  uint64_t _mask; // Lanemask for coalesced and tiled partitioned group types,
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

 public:
  // Total number of threads in the thread group, and this serves the purpose
  // for all derived cooperative group types since their `size` is directly
  // saved during the construction
  __CG_QUALIFIER__ uint32_t size() const {
    return _size;
  }
  // Rank of the calling thread within [0, size())
  __CG_QUALIFIER__ uint32_t thread_rank() const;
  // Is this cooperative group type valid?
  __CG_QUALIFIER__ bool is_valid() const;
  // synchronize the threads in the thread group
  __CG_QUALIFIER__ void sync() const;
};

/** \brief User exposed API interface to construct a generic `thread_group`
 *         object containing only the calling thread
 */
__CG_QUALIFIER__ thread_group
this_thread() {
  return thread_group(internal::cg_coalesced, (uint32_t)1, internal::lanemask_eq());
}

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
      : thread_group(internal::cg_multi_grid, size) { }

 public:
  // Number of invocations participating in this multi-grid group. In other
  // words, the number of GPUs
	__CG_QUALIFIER__ uint32_t num_grids() {
    return internal::multi_grid::num_grids();
  }
  // Rank of this invocation. In other words, an ID number within the range
  // [0, num_grids()) of the GPU, this kernel is running on
	__CG_QUALIFIER__ uint32_t grid_rank() {
    return internal::multi_grid::grid_rank();
  }
  __CG_QUALIFIER__ uint32_t thread_rank() const {
    return internal::multi_grid::thread_rank();
  }
  __CG_QUALIFIER__ bool is_valid() const {
    return internal::multi_grid::is_valid();
  }
  __CG_QUALIFIER__ void sync() const {
    internal::multi_grid::sync();
  }
};

/** \brief User exposed API interface to construct multi-grid cooperative
 *         group type object - `multi_grid_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `multi_grid_group`. Instead, he should construct it through this
 *           API function
 */
__CG_QUALIFIER__ multi_grid_group
this_multi_grid() {
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
  explicit __CG_QUALIFIER__ grid_group(uint32_t size)
      : thread_group(internal::cg_grid, size) { }

 public:
  __CG_QUALIFIER__ uint32_t thread_rank() const {
    return internal::grid::thread_rank();
  }
  __CG_QUALIFIER__ bool is_valid() const {
    return internal::grid::is_valid();
  }
  __CG_QUALIFIER__ void sync() const {
    internal::grid::sync();
  }
};

/** \brief User exposed API interface to construct grid cooperative group type
 *         object - `grid_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `multi_grid_group`. Instead, he should construct it through this
 *           API function
 */
__CG_QUALIFIER__ grid_group
this_grid() {
  return grid_group(internal::grid::size());
}

/** \brief The workgroup (thread-block in CUDA terminology) cooperative group
 *         type
 *
 *  \details Represents an intra-workgroup cooperative group type where the
 *           participating threads within the group are exctly the same threads
 *           which are participated in the currently executing `workgroup`
 */
class thread_block : public thread_group {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  friend __CG_QUALIFIER__ thread_block this_thread_block();

 protected:
  // Construct a workgroup thread group (through the API this_thread_block())
  explicit __CG_QUALIFIER__ thread_block(uint32_t size)
      : thread_group(internal::cg_workgroup, size) { }

 public:
  // 3-dimensional block index within the grid
  __CG_QUALIFIER__ dim3 group_index() {
    return internal::workgroup::group_index();
  }
  // 3-dimensional thread index within the block
  __CG_QUALIFIER__ dim3 thread_index() {
    return internal::workgroup::thread_index();
  }
  __CG_QUALIFIER__ uint32_t thread_rank() const {
    return internal::workgroup::thread_rank();
  }
  __CG_QUALIFIER__ bool is_valid() const {
    return internal::workgroup::is_valid();
  }
  __CG_QUALIFIER__ void sync() const {
    internal::workgroup::sync();
  }
};

/** \brief User exposed API interface to construct workgroup cooperative
 *         group type object - `thread_block`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `thread_block`. Instead, he should construct it through this API
 *           function
 */
__CG_QUALIFIER__ thread_block
this_thread_block() {
  return thread_block(internal::workgroup::size());
}

/** \brief The coalesced cooperative group type
 *
 *  \details Represents an intra-workgroup cooperative group type where each
 *           thread belongs to this group is an `active` lane of currently
 *           executing wavefront, and as such the allowed group size is only
 *           power of two and upto wavefront size.
 */
class coalesced_group : public thread_group {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  friend __CG_QUALIFIER__ coalesced_group coalesced_threads();
  friend __CG_QUALIFIER__ thread_group tiled_partition(
      const thread_group& parent, uint32_t tile_sz);

 protected:
  // Construct a coalesced thread group (through the API coalesced_threads())
  __CG_QUALIFIER__ coalesced_group(uint32_t size, uint64_t mask)
      : thread_group(internal::cg_coalesced, size, mask) { }

  __CG_QUALIFIER__ coalesced_group(internal::group_type type, uint32_t size,
                                   uint64_t mask)
      : thread_group(type, size, mask) { }

 public:
  __CG_QUALIFIER__ uint32_t thread_rank() const {
    return internal::coalesced::thread_rank(_mask);
  }
  __CG_QUALIFIER__ bool is_valid() const {
    return internal::coalesced::is_valid();
  }
  __CG_QUALIFIER__ void sync() const {
    internal::coalesced::sync();
  }
};

/** \brief User exposed API interface to construct coalesced cooperative group
 *         type object - `coalesced_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `coalesced_group`. Instead, he should construct it through this API
 *           function
 */
__CG_QUALIFIER__ coalesced_group
coalesced_threads() {
  uint64_t mask = internal::activemask();
  uint32_t size = internal::popcll(mask);
  return coalesced_group(size, mask);
}

/** \brief Templated (static) tiled partitioning of `exiting` intra-workgroup
 *         (and only intra-workgroup) cooperative group types
 *
 *  \details This class acts as a common base class definittion for all the
 *           templated tiled partitioned cooperative group type with varying
 *           tile sizes. The number of participating threads within this
 *           templated tiled partitioned cooperative group is specified at
 *           compile time as a template parameter. The allowed group size is
 *           only power of two and upto wavefront size. All the machinery of the
 *           tmplated partitioned cooperative group type implementation goes
 *           within this base class. The type derived from this base class is
 *           explicitly instantiated for all the allowed valid tile sizes.
 */
template <uint32_t tile_sz>
class thread_block_tile_base : public coalesced_group {
 protected:
  explicit __CG_QUALIFIER__ thread_block_tile_base(uint64_t mask)
      : coalesced_group(internal::cg_tiled_partition_static,
                        internal::popcll(mask), mask) { }

  __CG_QUALIFIER__ thread_block_tile_base(internal::group_type type, uint64_t mask)
      : coalesced_group(type, internal::popcll(mask), mask) { }
};

/** \brief The tiled partitioned cooperative group type
 *
 *  \details These classes derive the class `thread_block_tile_base` and they
 *           are explicitly instantiated for the allowed tile sizes
 */
template <>
class thread_block_tile<1> : public thread_block_tile_base<1> {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  template <uint32_t tile_sz>
  friend __CG_QUALIFIER__ thread_block_tile<tile_sz> tiled_partition(
      const thread_group& parent);

 protected:
  explicit __CG_QUALIFIER__ thread_block_tile(uint64_t mask)
    : thread_block_tile_base(mask) { }

  __CG_QUALIFIER__ thread_block_tile(internal::group_type type, uint64_t mask)
    : thread_block_tile_base(type, mask) { }
};

template <>
class thread_block_tile<2> : public thread_block_tile_base<2> {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  template <uint32_t tile_sz>
  friend __CG_QUALIFIER__ thread_block_tile<tile_sz> tiled_partition(
      const thread_group& parent);

 protected:
  explicit __CG_QUALIFIER__ thread_block_tile(uint64_t mask)
    : thread_block_tile_base(mask) { }

  __CG_QUALIFIER__ thread_block_tile(internal::group_type type, uint64_t mask)
    : thread_block_tile_base(type, mask) { }
};

template <>
class thread_block_tile<4> : public thread_block_tile_base<4> {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  template <uint32_t tile_sz>
  friend __CG_QUALIFIER__ thread_block_tile<tile_sz> tiled_partition(
      const thread_group& parent);

 protected:
  explicit __CG_QUALIFIER__ thread_block_tile(uint64_t mask)
    : thread_block_tile_base(mask) { }

  __CG_QUALIFIER__ thread_block_tile(internal::group_type type, uint64_t mask)
    : thread_block_tile_base(type, mask) { }
};

template <>
class thread_block_tile<8> : public thread_block_tile_base<8> {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  template <uint32_t tile_sz>
  friend __CG_QUALIFIER__ thread_block_tile<tile_sz> tiled_partition(
      const thread_group& parent);

 protected:
  explicit __CG_QUALIFIER__ thread_block_tile(uint64_t mask)
    : thread_block_tile_base(mask) { }

  __CG_QUALIFIER__ thread_block_tile(internal::group_type type, uint64_t mask)
    : thread_block_tile_base(type, mask) { }
};

template <>
class thread_block_tile<16> : public thread_block_tile_base<16> {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  template <uint32_t tile_sz>
  friend __CG_QUALIFIER__ thread_block_tile<tile_sz> tiled_partition(
      const thread_group& parent);

 protected:
  explicit __CG_QUALIFIER__ thread_block_tile(uint64_t mask)
    : thread_block_tile_base(mask) { }

  __CG_QUALIFIER__ thread_block_tile(internal::group_type type, uint64_t mask)
    : thread_block_tile_base(type, mask) { }
};

template <>
class thread_block_tile<32> : public thread_block_tile_base<32> {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  template <uint32_t tile_sz>
  friend __CG_QUALIFIER__ thread_block_tile<tile_sz> tiled_partition(
      const thread_group& parent);

 protected:
  explicit __CG_QUALIFIER__ thread_block_tile(uint64_t mask)
    : thread_block_tile_base(mask) { }

  __CG_QUALIFIER__ thread_block_tile(internal::group_type type, uint64_t mask)
    : thread_block_tile_base(type, mask) { }
};

template <>
class thread_block_tile<64> : public thread_block_tile_base<64> {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  template <uint32_t tile_sz>
  friend __CG_QUALIFIER__ thread_block_tile<tile_sz> tiled_partition(
      const thread_group& parent);

 protected:
  explicit __CG_QUALIFIER__ thread_block_tile(uint64_t mask)
    : thread_block_tile_base(mask) { }

  __CG_QUALIFIER__ thread_block_tile(internal::group_type type, uint64_t mask)
    : thread_block_tile_base(type, mask) { }
};

/** \brief User exposed API interface to construct new coalesced (dynamic) tiled
 *         partitioned group from an `exiting` intra-workgroup (and only
 *         intra-workgroup) cooperative group types
 */
__CG_QUALIFIER__ thread_group
tiled_partition(const thread_group& parent, uint32_t tile_sz) {
  if (!internal::is_tile_size_valid(tile_sz, parent.size())) {
    //TODO(mahesha) Do we need to abort the thread here?
    return thread_group(internal::cg_invalid, 0);
  }

  uint64_t mask = 0;
  internal::group_type gtype = internal::cg_invalid;
  switch (parent._type) {
    case internal::cg_workgroup:
    case internal::cg_tiled_partition_dynamic:
    case internal::cg_tiled_partition_static: {
      mask = internal::get_new_tiled_mask(tile_sz);
      gtype = internal::cg_tiled_partition_dynamic;
      break;
    }
    case internal::cg_coalesced: {
      //TODO(mahesha):
      break;
    }
  }

  return coalesced_group(gtype, internal::popcll(mask), mask);
}

/** \brief User exposed API interface to construct new templated (dynamic) tiled
 *         partitioned group from an `exiting` intra-workgroup (and only
 *         intra-workgroup) cooperative group types
 */
template <uint32_t tile_sz>
__CG_QUALIFIER__ thread_block_tile<tile_sz>
tiled_partition(const thread_group& parent) {
  if (!internal::is_tile_size_valid(tile_sz, parent.size())) {
    //TODO(mahesha) Do we need to abort the thread here?
    return thread_block_tile<tile_sz>(internal::cg_invalid, 0);
  }

  uint64_t mask = 0;
  internal::group_type gtype = internal::cg_invalid;
  switch (parent._type) {
    case internal::cg_workgroup:
    case internal::cg_tiled_partition_dynamic:
    case internal::cg_tiled_partition_static: {
      mask = internal::get_new_tiled_mask(tile_sz);
      gtype = internal::cg_tiled_partition_static;
      break;
    }
    case internal::cg_coalesced: {
      //TODO(mahesha):
      break;
    }
  }

  return thread_block_tile<tile_sz>(gtype, mask);
}

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
    case internal::cg_coalesced:
    case internal::cg_tiled_partition_dynamic:
    case internal::cg_tiled_partition_static: {
      return (static_cast<const coalesced_group*>(this)->thread_rank());
    }
    default: {
      return 0; //TODO(mahesha)
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
    case internal::cg_coalesced:
    case internal::cg_tiled_partition_dynamic:
    case internal::cg_tiled_partition_static: {
      return (static_cast<const coalesced_group*>(this)->is_valid());
    }
    default: {
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
    case internal::cg_coalesced:
    case internal::cg_tiled_partition_dynamic:
    case internal::cg_tiled_partition_static: {
      static_cast<const coalesced_group*>(this)->sync();
      break;
    }
  }
}

} // namespace cooperative_groups

#endif // (__cplusplus) && (__HIP_VDI__)
#endif // HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_H
