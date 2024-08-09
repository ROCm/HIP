.. meta::
  :description: HIP cooperative groups tutorial
  :keywords: AMD, ROCm, HIP, cooperative groups, tutorial

*******************************************************************************
Cooperative groups
*******************************************************************************

This tutorial demonstrates the basic concepts of cooperative groups in the HIP (Heterogeneous-computing Interface for Portability) programming model and the most essential tooling supporting it. This topic also reviews the commonalities of heterogeneous APIs. Familiarity with the C/C++ compilation model and the language is assumed.

Prerequisites
=============

To follow this tutorial, you'll need properly installed drivers and a HIP compiler toolchain to compile your code. Because ROCm HIP supports compiling and running on Linux and Microsoft Windows with AMD and NVIDIA GPUs, review the HIP development package installation before starting this tutorial. For more information, see :doc:`/install/install`.

Simple HIP Code
===============

To become familiar with heterogeneous programming, review the :doc:`SAXPY tutorial <saxpy>` and the first HIP code subsection. Compiling is also described in that tutorial.

Tiled partition
===============

You can use tiled partition to calculate the sum of ``partition_size`` length sequences and the sum of ``result_size``/ ``BlockSize`` length sequences. The host-side reference implementation is the following:

.. code-block:: cpp

  // Host-side function to perform the same reductions as executed on the GPU
  std::vector<unsigned int> ref_reduced(const unsigned int        partition_size,
                                        std::vector<unsigned int> input)
  {
      const unsigned int        input_size  = input.size();
      const unsigned int        result_size = input_size / partition_size;
      std::vector<unsigned int> result(result_size);

      for(unsigned int i = 0; i < result_size; i++)
      {
          unsigned int partition_result = 0;
          for(unsigned int j = 0; j < partition_size; j++)
          {
              partition_result += input[partition_size * i + j];
          }
          result[i] = partition_result;
      }

      return result;
  }

Device-side code
----------------

To calculate the sum of the sets of numbers, the tutorial uses the shared memory-based reduction on the device side. The warp level intrinsics usage is not covered in this tutorial, unlike in the :doc:`reduction tutorial. <reduction>` ``x`` input variable is a shared pointer, which needs to be synchronized after every value change. The ``thread_group`` input parameter can be ``thread_block_tile`` or ``thread_block`` because the ``thread_group`` is the parent class of these types. The ``val`` are the numbers to calculate the sum of. The returned results of this function return the final results of the reduction on thread ID 0 of the ``thread_group``, and for every other thread, the function results are 0.

.. code-block:: cuda

  /// \brief Summation of `unsigned int val`'s in `thread_group g` using shared memory `x`
  __device__ unsigned int reduce_sum(thread_group g, unsigned int* x, unsigned int val)
  {
      // Rank of this thread in the group
      const unsigned int group_thread_id = g.thread_rank();

      // We start with half the group size as active threads
      // Every iteration the number of active threads halves, until we processed all values
      for(unsigned int i = g.size() / 2; i > 0; i /= 2)
      {
          // Store value for this thread in a shared, temporary array
          x[group_thread_id] = val;

          // Synchronize all threads in the group
          g.sync();

          // If our thread is still active, sum with its counterpart in the other half
          if(group_thread_id < i)
          {
              val += x[group_thread_id + i];
          }

          // Synchronize all threads in the group
          g.sync();
      }

      // Only the first thread returns a valid value
      if(g.thread_rank() == 0)
          return val;
      else
          return 0;
  }

The ``reduce_sum`` device function is reused to calculate the block and custom
partition sum of the input numbers. The kernel has three sections:

1. Initialization of the reduction function variables.
2. The reduction of thread block and store the results in global memory.
3. The reduction of custom partition and store the results in global memory.

1. Initialization of the reduction function variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this code section, the shared memory is declared, the thread_block_group and
custom_partition are defined, and the input variables are loaded from global
memory.

.. code-block:: cuda

  // threadBlockGroup consists of all threads in the block
  thread_block thread_block_group = this_thread_block();

  // Workspace array in shared memory required for reduction
  __shared__ unsigned int workspace[2048];

  unsigned int output;

  // Input to reduce
  const unsigned int input = d_vector[thread_block_group.thread_rank()];

  // ...

  // Every custom_partition group consists of 16 threads
  thread_block_tile<PartitionSize> custom_partition
          = tiled_partition<PartitionSize>(thread_block_group);


2. The reduction of thread block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this code section, the sum is calculated on ``thread_block_group`` level, then the results are stored in global memory.

.. code-block:: cuda

  // Perform reduction
  output = reduce_sum(thread_block_group, workspace, input);

  // Only the first thread returns a valid value
  if(thread_block_group.thread_rank() == 0)
  {
      d_block_reduced_vector[0] = output;
  }

3. The reduction of custom partition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this code section, the sum is calculated on the custom partition level, then the results are stored in global memory. The custom partition is a partial block of the thread block, it means the reduction calculates on a shorter sequence of input numbers than at the ``thread_block_group`` case.

.. code-block:: cuda

  // Perform reduction
  output = reduce_sum(custom_partition, &workspace[group_offset], input);

  // Only the first thread in each partition returns a valid value
  if(custom_partition.thread_rank() == 0)
  {
      const unsigned int partition_id          = thread_block_group.thread_rank() / PartitionSize;
      d_partition_reduced_vector[partition_id] = output;
  }

Host-side code
--------------

On the host-side, the following steps are done in the example:

1. Confirm the cooperative group support on AMD GPUs.
2. Initialize the cooperative group configuration.
3. Allocate and copy input to global memory.
4. Launch the cooperative kernel.
5. Save the results from global memory.
6. Free the global memory.

Only the first, second and fourth steps are important from the cooperative groups aspect, that's why those steps are detailed further.

1. Confirm the cooperative group support on AMD GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not all AMD GPUs support cooperative groups. You can confirm support with the following code:

.. code-block:: cpp

  #ifdef __HIP_PLATFORM_AMD__
      int device               = 0;
      int supports_coop_launch = 0;
      // Check support
      // Use hipDeviceAttributeCooperativeMultiDeviceLaunch when launching across multiple devices
      HIP_CHECK(hipGetDevice(&device));
      HIP_CHECK(
          hipDeviceGetAttribute(&supports_coop_launch, hipDeviceAttributeCooperativeLaunch, device));
      if(!supports_coop_launch)
      {
          std::cout << "Skipping, device " << device << " does not support cooperative groups"
                    << std::endl;
          return 0;
      }
  #endif

2. Initialize the cooperative group configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the example, there is only one block in the grid, and the ``threads_per_block`` must be dividable with ``partition_size``.

.. code-block:: cpp

  // Number of blocks to launch.
  constexpr unsigned int num_blocks = 1;

  // Number of threads in each kernel block.
  constexpr unsigned int threads_per_block = 64;

  // Total element count of the input vector.
  constexpr unsigned int size = num_blocks * threads_per_block;

  // Total elements count of a tiled_partition.
  constexpr unsigned int partition_size = 16;

  // Total size (in bytes) of the input vector.
  constexpr size_t size_bytes = sizeof(unsigned int) * size;

  static_assert(threads_per_block % partition_size == 0,
                "threads_per_block must be a multiple of partition_size");

4. Launch the kernel
~~~~~~~~~~~~~~~~~~~~

The kernel launch is done with the ``hipLaunchCooperativeKernel`` of the  cooperative groups API.

.. code-block:: cpp

  void* params[] = {&d_vector, &d_block_reduced, &d_partition_reduced};
  // Launching kernel from host.
  HIP_CHECK(hipLaunchCooperativeKernel(vector_reduce_kernel<partition_size>,
                                       dim3(num_blocks),
                                       dim3(threads_per_block),
                                       params,
                                       0,
                                       hipStreamDefault));\

  // Check if the kernel launch was successful.
  HIP_CHECK(hipGetLastError());

Conclusion
==========

With cooperative groups, you can easily use custom partitions to create custom tiles for custom solutions. You can find the complete code at `cooperative groups ROCm example. <https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/cooperative_groups>`_
