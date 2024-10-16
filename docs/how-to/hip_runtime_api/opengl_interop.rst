.. meta::
   :description: HIP provides an OpenGL interoperability API that allows efficient data sharing between HIP's computing power and OpenGL's graphics rendering.
   :keywords: AMD, ROCm, HIP, OpenGL, interop, interoperability

**********************************************************
OpenGL interoperability
**********************************************************

Mapping
=======

Mapping resources
-----------------

The initial step in HIP--OpenGL interoperation involves mapping OpenGL resources, such as buffers and textures, for HIP access. This mapping process enables HIP to utilize these resources directly, bypassing the need for data transfers between the CPU and GPU. Specific HIP runtime API functions are employed to map the resources, making them available for HIP kernels' computations. This step is crucial for applications requiring large dataset processing using GPU computational power. By creating a bridge between OpenGL and HIP, mapping significantly enhances data handling efficiency. Intermediate data copies are eliminated, resulting in expedited data processing and rendering. Consequently, mapping resources is essential for smooth interoperation.

Getting mapped pointers
-----------------------

Following the mapping of resources, the next task is obtaining the device-accessible address of the OpenGL resource. HIP API functions are utilized to retrieve pointers to these mapped resources. These pointers allow HIP kernels to directly access and manipulate the data within OpenGL resources. The process entails querying the mapped resource for its device address, which can then be used in HIP kernels for reading and writing operations. Direct access through the device-addressable pointer reduces overhead associated with data movement and enhances overall performance. The integration of the device address facilitates seamless data handling and sharing between HIP and OpenGL.

Unmapping resources
-------------------

Upon completion of the necessary computations and data manipulations, resources must be unmapped to release HIP access. HIP API functions are employed to unmap the resources, ensuring proper management and availability for future use. Unmapping resources signals the end of their usage by HIP, maintaining the integrity and availability of these resources. This step prevents unnecessary retention of resources, thereby freeing up GPU memory. Effective resource lifecycle management is achieved through the unmapping process, contributing to system stability and efficiency. Properly concluding the interoperation process necessitates unmapping resources.

Registering resources
=====================

To enable HIP--OpenGL interoperation, registering OpenGL resources with HIP is required. This process creates corresponding HIP graphics resources, utilized in HIP kernels. Registration is accomplished using HIP runtime API functions, which take OpenGL resources and generate their HIP representations. These HIP graphics resources can then be mapped, accessed, and manipulated within HIP. This preparatory step ensures resources are properly identified and managed within the HIP environment. Dual accessibility of resources in both OpenGL and HIP contexts is achieved through registration. The registration of resources establishes the foundation for efficient and integrated data handling in HIP--OpenGL interoperation.

Examples
========

Two examples are presented with mapping resources, getting pointers -- directly or with arrays -- and unmapping resources with a OpenGL buffer registration.

.. tab-set::

    .. tab-item:: with mapped pointer

        .. code-block:: cpp
            :emphasize-lines: 21-24

            #include <hip/hip_runtime.h>
            #include <hip/hip_runtime_api.h>
            #include <GL/glew.h>
            #include <GL/gl.h>

            int main()
            {
                // Initialize OpenGL and create a buffer
                GLuint buffer;
                glGenBuffers(1, &buffer);
                glBindBuffer(GL_ARRAY_BUFFER, buffer);
                glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

                // Register the OpenGL buffer with HIP
                hipGraphicsResource* resource;
                hipGraphicsGLRegisterBuffer(&resource, buffer, hipGraphicsRegisterFlagsNone);

                // Map the resource for access by HIP
                hipGraphicsMapResources(1, &resource, 0);

                // Obtain a pointer to the mapped resource
                void* devicePtr;
                size_t numBytes;
                hipGraphicsResourceGetMappedPointer(&devicePtr, &numBytes, resource);

                // Use devicePtr in HIP kernels...

                // Unmap the resources when done
                hipGraphicsUnmapResources(1, &resource, 0);

                // Cleanup OpenGL resources
                glDeleteBuffers(1, &buffer);

                return 0;
            }

    .. tab-item:: with mapped array

        .. code-block:: cpp

            :emphasize-lines: 20-22

            #include <hip/hip_runtime.h>
            #include <hip/hip_runtime_api.h>
            #include <GL/glew.h>
            #include <GL/gl.h>

            int main()
            {
                // Initialize OpenGL and create a buffer
                GLuint buffer;
                glGenBuffers(1, &buffer);
                glBindBuffer(GL_ARRAY_BUFFER, buffer);
                glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

                // Register the OpenGL buffer with HIP
                hipGraphicsResource* resource;
                hipGraphicsGLRegisterBuffer(&resource, buffer, hipGraphicsRegisterFlagsNone);

                // Map the resource for access by HIP
                hipGraphicsMapResources(1, &resource, 0);

                // Obtain a pointer to the mapped array
                hipArray* arrayPtr;
                hipGraphicsSubResourceGetMappedArray(&arrayPtr, resource, 0, 0);

                // Use arrayPtr in HIP kernels...

                // Unmap the resources when done
                hipGraphicsUnmapResources(1, &resource, 0);

                // Cleanup OpenGL resources
                glDeleteBuffers(1, &buffer);

                return 0;
            }

An other example is with mapping resources, getting pointers and unmapping resources with a OpenGL image registration.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>
    #include <GL/glew.h>
    #include <GL/gl.h>

    int main() {
        // Initialize OpenGL and create a texture
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

        // Register the OpenGL texture with HIP
        hipGraphicsResource* resource;
        hipGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, hipGraphicsRegisterFlagsNone);

        // Map the resource for access by HIP
        hipGraphicsMapResources(1, &resource, 0);

        // Obtain a pointer to the mapped array
        hipArray* arrayPtr;
        hipGraphicsSubResourceGetMappedArray(&arrayPtr, resource, 0, 0);

        // Use arrayPtr in HIP kernels...

        // Unmap the resources when done
        hipGraphicsUnmapResources(1, &resource, 0);

        // Cleanup OpenGL resources
        glDeleteTextures(1, &texture);

        return 0;
    }
