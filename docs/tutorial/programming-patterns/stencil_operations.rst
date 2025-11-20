.. meta::
  :description: Image convolution tutorial using HIP on AMD GPUs
  :keywords: AMD, ROCm, HIP, stencil operations, image convolution, GPU programming, data parallelism

*******************************************************************************
Stencil operations: Image convolution tutorial
*******************************************************************************

Stencil operations represent an important class of **embarrassingly parallel**
algorithms that are ideally suited for GPU acceleration. A stencil algorithm
iteratively updates data in an array based on a data item's adjacent cells,
making it a fundamental technique in various computational domains.

.. include:: ../prerequisites.rst

Applications of stencil operations
===================================

Stencil algorithms are commonly used in:

* **Physics simulations**: Modeling heat transfer, fluid dynamics, and wave
  propagation

* **Partial differential equations**: Numerical solutions to scientific
  computing problems

* **Image processing**: Convolutional operations for smoothing, sharpening, and
  edge detection

* **Convolutional Neural Networks**: A major building block in modern
  deep learning

By applying different image convolution kernels (not to be confused with GPU
kernels), these algorithms can smooth and sharpen image features and detect
edges effectively.

Image convolution
=================

An image convolution applies a small matrix (the **mask** or **filter kernel**)
to an input image. For each pixel :math:`(x, y)`, the output is computed as:

.. math::

   I'(x, y) = \sum_{i=-r}^{r} \sum_{j=-r}^{r} M(i, j) \cdot I(x + i, y + j)

where :math:`M(i, j)` is the mask coefficient, and :math:`r` is half the mask
width (assuming a square kernel).

Step by step description of the equation:

1. Center the mask over the current pixel
2. Multiply each mask value by the corresponding image pixel
3. Sum all the products
4. Store the result as the new pixel value

Dimensionality of stencils
--------------------------

Stencil computations extend beyond 2D image grids:

* **1D:** Signal filtering, time series processing  

* **2D:** Image processing, texture analysis  

* **3D:** Volume data, fluid flow, and physical field simulation  

This tutorial focuses on **2D image convolution**, the most common stencil
operation in visual and scientific computing.

The smoothing operation
-----------------------

The tutorial implements a **box blur** (uniform smoothing filter). Each
pixel’s new value is the average of its local neighborhood. This operation:

* Reduces noise by averaging local intensity variations.

* Acts as a low-pass filter, attenuating high-frequency components.

* Provides an ideal example of a stencil computation with uniform weights.

Two-dimensional grid architecture
==================================

The tutorial uses a two-dimensional grid that maps to the shape of the image, significantly
simplifying the implementation. This approach:

* Maps threads directly to pixel positions

* Simplifies coordinate calculations

* Enables intuitive spatial reasoning

* Aligns with the natural structure of images

Grid configuration
------------------

Rather than using a single integer to represent the size of the grid, the tutorial uses a
``dim3`` object containing three values to represent the number of block-based
work items per dimension:

* **x dimension**: Width of the image

* **y dimension**: Height of the image

* **z dimension**: Set to 1 for 2D problems

Complete implementation
=======================

Header and setup
----------------

.. code-block:: c++

    #include <hip/hip_runtime.h>
    #include <vector>
    #include "image.h"

The convolution kernel
----------------------

Here's the complete 2D convolution kernel for image smoothing:

.. code-block:: c++

    __global__ void conv2d(uint8_t *image, float *mask, int image_width,
                          int image_height, int mask_width, int mask_height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= image_width || y >= image_height) {
            return;
        }
        
        float sum = 0;
        for (int i = 0; i < mask_width; i++) {
            for (int j = 0; j < mask_height; j++) {
                // Calculate the coordinate of the pixel to read.
                int image_x = x + i - mask_width / 2;
                int image_y = y + j - mask_height / 2;
                
                // Do not read outside the image.
                if (image_x < 0 || image_x >= image_width || 
                    image_y < 0 || image_y >= image_height) {
                    continue;
                }
                
                // Accumulate the value of the pixel.
                int image_index = image_y * image_width + image_x;
                int mask_index = j * mask_width + i;
                sum += image[image_index] / 255.0f * mask[mask_index];
            }
        }
        
        int image_index = y * image_width + x;
        image[image_index] = sum * 255;
    }

Thread identification in 2D
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To obtain thread IDs in both x and y dimensions:

.. code-block:: c++

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

This calculation combines:

* ``threadIdx.x`` and ``threadIdx.y``: Local thread coordinates within a block

* ``blockIdx.x`` and ``blockIdx.y``: Block coordinates within the grid

* ``blockDim.x`` and ``blockDim.y``: Block dimensions

Boundary checking
~~~~~~~~~~~~~~~~~

.. code-block:: c++

    if (x >= image_width || y >= image_height) {
        return;
    }

This ensures threads don't process pixels outside the image bounds. Threads that
exceed the image dimensions simply return without doing work.

Mask application loop
~~~~~~~~~~~~~~~~~~~~~

The nested loops iterate over the mask dimensions:

.. code-block:: c++

    for (int i = 0; i < mask_width; i++) {
        for (int j = 0; j < mask_height; j++) {
            // Process each mask element
        }
    }

Coordinate calculation
~~~~~~~~~~~~~~~~~~~~~~

For each position in the mask, calculate the corresponding image coordinate:

.. code-block:: c++

    int image_x = x + i - mask_width / 2;
    int image_y = y + j - mask_height / 2;

The ``- mask_width / 2`` and ``- mask_height / 2`` center the mask over the
current pixel.

Edge handling
~~~~~~~~~~~~~

.. code-block:: c++

    if (image_x < 0 || image_x >= image_width || 
        image_y < 0 || image_y >= image_height) {
        continue;
    }

This prevents reading outside the image boundaries. When the mask extends beyond
the image edge, the code simply skips those pixels (continue to the next
iteration).

Accumulation
~~~~~~~~~~~~

.. code-block:: c++

    int image_index = image_y * image_width + image_x;
    int mask_index = j * mask_width + i;
    sum += image[image_index] / 255.0f * mask[mask_index];

For each mask position:

1. Calculate the flattened array index for the image pixel
2. Calculate the flattened array index for the mask value
3. Normalize the pixel value (divide by 255 to get 0-1 range)
4. Multiply by the mask weight and accumulate

Writing the result
~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    int image_index = y * image_width + x;
    image[image_index] = sum * 255;

After processing all mask positions, write the accumulated result back to the
output image, scaling back to the 0-255 range.

Host code implementation
------------------------

Main function setup
~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    int main() {
        int width, height, channels;
        static const int maskWidth = 200;
        static const int maskHeight = 200;
        std::vector<float> mask(maskWidth * maskHeight * channels);
        
        // Initialize mask with uniform averaging weights
        for (int i = 0; i < maskWidth * maskHeight; ++i) {
            mask[i] = 1.0f / maskWidth / maskHeight / channels;
        }
        
        // Load an image from disk (implementation not shown)

Mask initialization
~~~~~~~~~~~~~~~~~~~

The mask is initialized with uniform weights that sum to 1.0:

* Each element is ``1.0 / (maskWidth * maskHeight * channels)``

* This creates an averaging filter

* When applied, it produces a smoothing (blurring) effect

Memory allocation and data transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

        // Allocate GPU memory and copy data to the GPU.
        uint8_t *d_image;
        float *d_mask;
        hipMalloc(&d_image, width * height * channels * sizeof(uint8_t));
        hipMalloc(&d_mask, maskWidth * maskHeight * channels * sizeof(float));
        
        hipMemcpy(d_image, image, width * height * channels * sizeof(uint8_t),
                  hipMemcpyHostToDevice);
        hipMemcpy(d_mask, mask.data(),
                  maskWidth * maskHeight * channels * sizeof(float),
                  hipMemcpyHostToDevice);

Grid configuration and kernel launch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

        // Calculate grid size and launch the kernel.
        dim3 block_size = {16, 16, 1};
        dim3 grid_size = {(width + block_size.x - 1) / block_size.x,
                         (height + block_size.y - 1) / block_size.y, 1};
        
        conv2d<<<grid_size, block_size>>>(d_image, d_mask, width, height,
                                          maskWidth, maskHeight);
        hipDeviceSynchronize();

**Grid size calculation:**

* :code:`block_size`: 16 × 16 threads per block (256 threads total).

* :code:`grid_size`: Calculated to cover the entire image.

* The ``(width + block_size.x - 1) / block_size.x`` formula ensures there are
  enough blocks to cover all pixels, rounding up.

Retrieving results and cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

        // Copy the data back to the host.
        hipMemcpy(image, d_image, width * height * channels * sizeof(uint8_t),
                  hipMemcpyDeviceToHost);
        
        // Store the image to disk (implementation not shown)
        
        hipFree(d_image);
        hipFree(d_mask);
        
        return 0;
    }

Performance considerations
==========================

For high-resolution images for example 4096×4096 pixels, the GPU acceleration
provides:

* **Massive parallelism:** Tens of thousands of concurrent threads.

* **High throughput:** Leveraging GPU memory bandwidth and computational density.

* **Scalability:** Linear speedup with increased SM occupancy.

Typical speedups over CPU implementations range from **10× to 100×**, depending
on image size, mask complexity, and GPU architecture.

Memory access patterns
-----------------------

Image convolution requires repeated access to neighboring pixels, leading to
non-coalesced memory transactions. Optimizing memory access is essential:

* Non-coalesced memory accesses (not all accesses are contiguous).

* Repeated reads of the same pixels by adjacent threads.

* Potential for optimization using shared memory (advanced technique).

Best practices
==============

1. **Handle boundaries carefully**

   Conditional branches in edge regions can cause wavefronts or warps to execute
   serially. Prefer approaches that avoid per-thread branching, including:

     * Pre-clamping coordinates to valid index ranges

     * Padding or haloing input images so all threads operate on valid data

     * Using hardware boundary modes (such as texture sampling modes on RDNA) to
       offload boundary handling

   These techniques help maintain high SIMD lane utilization across the entire
   grid.

2. **Center your stencil properly**

   Compute the stencil origin using mask_width / 2 (or the equivalent for
   rectangular masks) to ensure correct alignment between the input data and the
   mask coefficients. This prevents off-by-one misalignment that can propagate
   as spatial artifacts.

3. **Select mask sizes based on compute–memory tradeoffs**

   Larger kernels increase arithmetic intensity but also expand the set of
   neighbor loads per output element. Balance mask dimensions with available
   bandwidth, register pressure, and shared-memory capacity, particularly when
   implementing separable or multi-pass stencils.

4. **Normalize properly**

   Ensure mask weights sum to the intended normalization constant, commonly 1.0
   for averaging operations. When using integer or half-precision paths, verify
   scaling behavior to avoid overflow or unintended bias.

5. **Consider edge strategies**

   Adopt a clear policy for pixels whose neighborhoods extend outside the valid
   domain. Options include skipping output generation, clamping to the nearest
   valid coordinate, wrapping coordinates, or mirroring.

Conclusion
==========

Stencil operations are a fundamental pattern in GPU computing that enables
efficient parallel processing of spatially-dependent data. The 2D convolution
example demonstrates:

* How to structure kernels for stencil patterns

* Proper boundary handling for neighborhood operations

* Effective use of 2D thread grids that map naturally to image structure

* Memory access patterns for adjacent data elements

By understanding stencil operations, developers can implement a wide range of
image processing algorithms, scientific simulations, and deep learning
operations on GPUs. The patterns demonstrated here extend beyond image
processing to any computational problem involving spatial relationships in
multi-dimensional data.

The key to successful stencil implementations is carefully managing boundary
conditions, ensuring correct coordinate calculations, and leveraging the GPU's
parallel architecture to process many independent stencil operations
simultaneously.
