/* Copyright (c) 2012-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef CL_KERNEL_H_
#define CL_KERNEL_H_

struct clk_builtins_t;

// This must be a multiple of sizeof(cl_ulong16)
#define __CPU_SCRATCH_SIZE 128

#define CLK_PRIVATE_MEMORY_SIZE (16 * 1024)

struct clk_thread_info_block_t {
  // Warning!  The size of this struct needs to be a multiple
  // of 16 when compiling 64 bit

  struct clk_builtins_t const* builtins;
  void* local_mem_base;
  void* local_scratch;
  const void* table_base;
  size_t pad;

  uint work_dim;
  size_t global_offset[4]; /*dim0,dim1,dim2,invalid(dim<0||dim>2)*/
  size_t global_size[4];   /*dim0,dim1,dim2,invalid(dim<0||dim>2)*/

  size_t enqueued_local_size[4];
  size_t local_size[4]; /*dim0,dim1,dim2,invalid(dim<0||dim>2)*/
  size_t local_id[4];   /*dim0,dim1,dim2,invalid(dim<0||dim>2)*/
  size_t group_id[4];   /*dim0,dim1,dim2,invalid(dim<0||dim>2)*/
};

typedef enum clk_value_type_t {
  T_VOID,
  T_CHAR,
  T_SHORT,
  T_INT,
  T_LONG,
  T_FLOAT,
  T_DOUBLE,
  T_POINTER,
  T_CHAR2,
  T_CHAR3,
  T_CHAR4,
  T_CHAR8,
  T_CHAR16,
  T_SHORT2,
  T_SHORT3,
  T_SHORT4,
  T_SHORT8,
  T_SHORT16,
  T_INT2,
  T_INT3,
  T_INT4,
  T_INT8,
  T_INT16,
  T_LONG2,
  T_LONG3,
  T_LONG4,
  T_LONG8,
  T_LONG16,
  T_FLOAT2,
  T_FLOAT3,
  T_FLOAT4,
  T_FLOAT8,
  T_FLOAT16,
  T_DOUBLE2,
  T_DOUBLE3,
  T_DOUBLE4,
  T_DOUBLE8,
  T_DOUBLE16,
  T_SAMPLER,
  T_SEMA,
  T_STRUCT,
  T_QUEUE,
  T_PAD
} clk_value_type_t;

typedef enum clk_address_space_t {
  A_PRIVATE,
  A_LOCAL,
  A_CONSTANT,
  A_GLOBAL,
  A_REGION
} clk_address_space_t;

// kernel arg access qualifier and type qualifier
typedef enum clk_arg_qualifier_t {
  Q_NONE = 0,

  // for image type only, access qualifier
  Q_READ = 1,
  Q_WRITE = 2,

  // for pointer type only
  Q_CONST = 4,  // pointee
  Q_RESTRICT = 8,
  Q_VOLATILE = 16,  // pointee
  Q_PIPE = 32       // pipe

} clk_arg_qualifier_t;

#pragma pack(push, 4)
struct clk_parameter_descriptor_t {
  clk_value_type_t type;
  clk_address_space_t space;
  uint qualifier;
  const char* name;
};
#pragma pack(pop)

//#define CLK_LOCAL_MEM_FENCE  (1 << 0)
//#define CLK_GLOBAL_MEM_FENCE (1 << 1)

struct clk_builtins_t {
  /* Synchronization functions */
  void (*barrier_ptr)(cl_mem_fence_flags flags);

  /* AMD Only builtins: FIXME_lmoriche (extension) */
  void* reserved;
  int (*printf_ptr)(const char* format, ...);
};

enum clk_natures_t { KN_HAS_BARRIER = 1 << 0, KN_WG_LEVEL = 1 << 1 };

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4200)
#endif

#if !defined(__OPENCL_VERSION__) || __OPENCL_VERSION__ >= 200

typedef struct clk_pipe_t {
  size_t read_idx;
  size_t write_idx;
  size_t end_idx;
  char padding[128 - 3 * sizeof(size_t)];
  char packets[];
} clk_pipe_t;

#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif /*CL_KERNEL_H_*/
