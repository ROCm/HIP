// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "nvidia_hip_fix.hpp"

#include "example_utils.hpp"

#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <hip/hip_gl_interop.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

/// \brief The number of triangles that the example's grid is in width.
constexpr uint32_t grid_width = 256;
/// \brief The number of triangles that the example's grid is in height.
constexpr uint32_t grid_height = 256;

/// \brief The OpenGL vertex shader that is used to render the triangles in this example.
/// The grid x- and y-positions are used to set the triangle coordinates in clip space.
/// The height value is passed on to the fragment shader.
constexpr const char* vertex_shader = R"(
#version 330 core

in float in_height;
in vec2 in_xy;

out float frag_height;

void main()
{
    gl_Position = vec4(in_xy, 0, 1);
    frag_height = in_height;
}
)";

/// \brief The OpenGL fragment shader that is used to render the triangles in this example.
/// The "height" value is used to shade the vertex. Its values are interpolated linearly
/// between the vertex and fragment shaders.
constexpr const char* fragment_shader = R"(
#version 330 core

in float frag_height;

void main()
{
    gl_FragColor = vec4(vec3(frag_height * 0.5 + 0.5), 1.0);
}
)";

/// \brief Initialize a GLFW window with initial dimensions.
GLFWwindow* create_window(const int initial_width, const int initial_height)
{
    /// [Sphinx-create-window]
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(initial_width,
                                          initial_height,
                                          "OpenGL-HIP interop example",
                                          nullptr,
                                          nullptr);
    if(window == nullptr)
    {
        std::cerr << "Failed to create GLFW window\n";
        std::exit(error_exit_code);
    }
    /// [Sphinx-create-window]
    return window;
}

/// \brief Select a HIP device that is compatible with the current OpenGL context.
/// \returns A HIP device-id that is capable of rendering the example. If no
///   suitable device is found, an error is printed and the program is exited.
int pick_hip_device()
{
    /// [Sphinx-pick device]
    unsigned int gl_device_count;
    int          hip_device;
    HIP_CHECK(
        hipGLGetDevices(&gl_device_count, &hip_device, 1, hipGLDeviceList::hipGLDeviceListAll));

    if(gl_device_count == 0)
    {
        std::cerr << "System has no OpenGL-capable HIP devices" << std::endl;
        std::exit(error_exit_code);
    }
    /// [Sphinx-pick device]

    return hip_device;
}

/// \brief Utility function to compile shader source into an OpenGL shader.
/// If the shader could not be compiled, this function prints the compile log
/// and exits the program.
/// \param type - The OpenGL shader type for this shader, for example
///   \p GL_VERTEX_SHADER or \p GL_FRAGMENT_SHADER.
/// \param source - The GLSL source code for the shader.
GLuint compile_shader(const GLenum type, const char* const source)
{
    const GLuint shader = glCreateShader(type);

    const GLint length = static_cast<GLint>(std::strlen(source));
    glShaderSource(shader, 1, &source, &length);
    glCompileShader(shader);

    GLint compile_status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);

    if(compile_status != GL_TRUE)
    {
        // Compiling failed, get the shader log and print it to the user.
        GLint log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<GLchar> log(log_length);
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        std::cerr << "Failed to compile shader:\n";
        std::cerr.write(log.data(), log.size()) << std::endl;
        std::exit(error_exit_code);
    }

    return shader;
}

/// \brief Utility function to compile and link a vertex and fragment shader into an OpenGL
/// shader program.
/// If the shaders could not be compiled, a log is printed and the program is exited.
/// \param vert_src - The GLSL source code for the shader program's vertex shader.
/// \param frag_src - The GLSL source code for the shader program's fragment shader.
GLuint compile_shader_program(const char* const vert_src, const char* const frag_src)
{
    const GLuint program = glCreateProgram();

    const GLuint vert = compile_shader(GL_VERTEX_SHADER, vert_src);
    const GLuint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src);

    glAttachShader(program, frag);
    glAttachShader(program, vert);

    glLinkProgram(program);

    GLint link_status;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if(link_status != GL_TRUE)
    {
        // Linking failed, get the program link log and print it to the user.
        GLint log_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<GLchar> log(log_length);
        glGetProgramInfoLog(program, log_length, nullptr, log.data());
        std::cerr << "Failed to link program:\n";
        std::cerr.write(log.data(), log.size()) << std::endl;
        std::exit(error_exit_code);
    }

    glDetachShader(program, frag);
    glDetachShader(program, vert);

    glDeleteShader(frag);
    glDeleteShader(vert);

    return program;
}

/// \brief This structure contains the OpenGL handles that this example uses to render the
/// triangle grid to the screen.
///
/// Three buffers are used to render the triangle grid, the color of which is determined by
/// a HIP compulation in \p simulator:
/// - One buffer contains the height of each triangle (rendered as color).
/// - One buffer holds the x- and y-coordinates for each of the corners of the triangle. Note: these
///   coordinates are unique, as the triangles that are made up from these points are defined by the
/// - Index buffer, that holds indices into the former two buffers to make up a list of triangles.
struct renderer
{
    /// The total number of vertices for the triangles.
    constexpr static size_t num_verts = grid_width * grid_height;
    /// The number of bytes in the x- and y-coordinates buffer. Each x/y coordinate is encoded as
    /// a pair of floats, which are stored in a packed  array-of-structures format: | x | y | x | y | ... |.
    constexpr static size_t grid_buffer_size = num_verts * sizeof(float) * 2;
    /// The number of bytes in the height buffer. Each height is encoded as a floating point value.
    /// This buffer will be shared with HIP, which is why these coordinates are
    /// stored in a separate buffer.
    constexpr static size_t height_buffer_size = num_verts * sizeof(float);

    /// The number of indices in the index buffer. Each triangle has 3 points, each square in the grid
    /// is made up of 2 triangles. There are (width - 1) by (height - 1) squares in the grid.
    constexpr static size_t num_indices = (grid_width - 1) * (grid_height - 1) * 3 * 2;
    /// The number of bytes in the index buffer. Each index is encoded as a 32-bit int.
    constexpr static size_t index_buffer_size = num_indices * sizeof(uint32_t);

    /// An OpenGL handle to a Vertex Array Object, which has the grid and height buffers
    /// bound to the corresponding attribute in the shader program (<tt>program</tt>) used for rendering.
    GLuint vao;

    /// Handle to the buffer that holds the indices for the triangles to render.
    GLuint index_buffer;

    /// Handle to the buffer that holds the x- and y-coordinates for each grid point.
    GLuint grid_buffer;

    /// Handle to the buffer that holds the heights each grid point. This buffer is shared with HIP.
    GLuint height_buffer;

    /// Handle to the OpenGL shader program that this example uses to render the triangles to the screen.
    GLuint program;

    /// Counters used to keep track of the rendering performance.
    uint32_t                                       fps_frame = 0;
    std::chrono::high_resolution_clock::time_point fps_start_time;

    /// \brief Initialize OpenGL rendering resources.
    renderer()
    {
        // Create a vertex array used to bind the attribute buffers.
        glGenVertexArrays(1, &this->vao);

        // Also generate the buffers in question.
        GLuint buffers[3];
        glGenBuffers(std::size(buffers), buffers);
        this->index_buffer  = buffers[0];
        this->grid_buffer   = buffers[1];
        this->height_buffer = buffers[2];

        // Compile the shader program used to render the triangles.
        this->program = compile_shader_program(vertex_shader, fragment_shader);

        // Upload the initial data to the buffers.
        this->initialize_buffer_data();

        // Set up the VAO by binding the height and grid buffers to the attribute locations
        // in the shader program.
        glBindVertexArray(this->vao);

        // Note - keep variable "in_height" in sync with shader.
        glBindBuffer(GL_ARRAY_BUFFER, this->height_buffer);
        const GLuint height_attrib = glGetAttribLocation(this->program, "in_height");
        glVertexAttribPointer(height_attrib, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(height_attrib);

        // Note - keep variable "in_xy" in sync with shader.
        const GLuint grid_attrib = glGetAttribLocation(this->program, "in_xy");
        glBindBuffer(GL_ARRAY_BUFFER, this->grid_buffer);
        glVertexAttribPointer(grid_attrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(grid_attrib);

        this->fps_start_time = std::chrono::high_resolution_clock::now();
    }

    renderer(const renderer&)            = delete;
    renderer& operator=(const renderer&) = delete;

    renderer(renderer&&)            = delete;
    renderer& operator=(renderer&&) = delete;

    ~renderer()
    {
        glDeleteProgram(this->program);
        GLuint buffers[] = {this->index_buffer, this->grid_buffer, this->height_buffer};
        glDeleteBuffers(std::size(buffers), buffers);
        glDeleteVertexArrays(1, &this->vao);
    }

    /// \brief Upload the initial values for each buffer to Vulkan.
    void initialize_buffer_data() const
    {
        // Initialize the height buffer.
        glBindBuffer(GL_ARRAY_BUFFER, this->height_buffer);
        // We do not need to fill it, as that is going to be done from HIP, but we
        // do need to allocate it from OpenGL. This is done simply by passing `nullptr` as
        // initial data pointer.
        // GL_DYNAMIC_DRAW is passed because this buffer is going to be updated every frame,
        // and is going to be used to hold vertex data for drawing - this may help the driver
        // to render more efficiently.
        glBufferData(GL_ARRAY_BUFFER, height_buffer_size, nullptr, GL_DYNAMIC_DRAW);

        // Initialize the grid buffer.
        {
            glBindBuffer(GL_ARRAY_BUFFER, this->grid_buffer);
            // Avoid having to allocate on host by allocating the buffer in OpenGL and then mapping it
            // into host-memory to initialize it.
            // This buffer is going to be initialized once and is going to be used for drawing,
            // so pass GL_STATIC_DRAW as usage hint.
            glBufferData(GL_ARRAY_BUFFER, grid_buffer_size, nullptr, GL_STATIC_DRAW);

            float* grid = reinterpret_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
            for(uint32_t y = 0; y < grid_height; ++y)
            {
                for(uint32_t x = 0; x < grid_width; ++x)
                {
                    *grid++ = (2.0f * x) / (grid_width - 1) - 1;
                    *grid++ = (2.0f * y) / (grid_height - 1) - 1;
                }
            }

            // Let OpenGL know that we are done with this buffer.
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }

        // Initialize the index buffer
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->index_buffer);
            // Similar as the grid buffer, this buffer is going to be initialized once and is then used
            // for drawing.
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_buffer_size, nullptr, GL_STATIC_DRAW);

            uint32_t* indices
                = reinterpret_cast<uint32_t*>(glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY));
            for(uint32_t y = 0; y < grid_height - 1; ++y)
            {
                for(uint32_t x = 0; x < grid_width - 1; ++x)
                {
                    *indices++ = (y + 0) * grid_width + (x + 0);
                    *indices++ = (y + 1) * grid_width + (x + 0);
                    *indices++ = (y + 0) * grid_width + (x + 1);
                    *indices++ = (y + 1) * grid_width + (x + 0);
                    *indices++ = (y + 1) * grid_width + (x + 1);
                    *indices++ = (y + 0) * grid_width + (x + 1);
                }
            }

            glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
        }
    }

    /// \brief Bind the OpenGL pipeline state for this renderer.
    void bind() const
    {
        glBindVertexArray(this->vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->index_buffer);
        glUseProgram(this->program);
    }

    /// \brief Draw the next frame to the window. This requires the render state be bound using
    /// <tt>bind</tt>.
    void draw()
    {
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);

        // Output a native performance measurement.
        ++this->fps_frame;
        const auto frame_time = std::chrono::high_resolution_clock::now();
        const auto time_diff  = frame_time - this->fps_start_time;
        if(time_diff > std::chrono::seconds{5})
        {
            const auto time_diff_sec
                = std::chrono::duration_cast<std::chrono::duration<float>>(time_diff).count();
            std::cout << "Average FPS (over " << double_precision(time_diff_sec, 2, true)
                      << " seconds): " << double_precision(this->fps_frame / time_diff_sec, 2, true)
                      << " (" << double_precision((time_diff_sec * 1000) / this->fps_frame, 2, true)
                      << " ms per frame, " << this->fps_frame << " frames)" << std::endl;
            this->fps_frame      = 0;
            this->fps_start_time = frame_time;
        }
    }
};

/// [Sphinx sinewave kernel start]
/// \brief The main HIP kernel for this example - computes a simple sine wave over a
/// 2-dimensional grid of points.
/// \param height_map - the grid of points to compute a sine wave for. It is expected to be
///   a \p grid_width by \p grid_height array packed into memory.(y on the inner axis).
/// \param time - The current time relative to the start of the program.
__global__ void sinewave_kernel(float* height_map, const float time)
{
    const float        freq = 10.f;
    const unsigned int x    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y    = blockIdx.y * blockDim.y + threadIdx.y;
    const float        u    = (2.f * x) / grid_width - 1.f;
    const float        v    = (2.f * y) / grid_height - 1.f;

    if(x < grid_width && y < grid_height)
    {
        height_map[x * grid_width + y] = sinf(u * freq + time) * cosf(v * freq + time);
    }
}
/// [Sphinx sinewave kernel end]

/// \brief This structure contains the HIP state and functionality used to advance the simulation.
/// Initializing a \p simulator fetches the OpenGL height buffer from the corresponding <tt>renderer</tt>,
/// and imports it as a HIP device pointer. This pointer is then passed to the simulation kernel
/// (<tt>sinewave_kernel</tt>), which updates the values in it. When <tt>renderer::draw</tt> is called,
/// the updated values are read from the buffer in OpenGL and used to render the triangle grid.
struct simulator
{
    /// The HIP stream used to advance the simulation. This must be created from an OpenGL-interop
    /// capable device, see <tt>pick_hip_device</tt>.
    hipStream_t hip_stream;
    /// A HIP graphics resource that is imported from the OpenGL height buffer to simulate.
    hipGraphicsResource_t hip_height_buffer;
    /// A device pointer to the height buffer, imported from the OPenGL height buffer.
    float* hip_height_ptr;

    /// The start time of the program, used for the simulation.
    std::chrono::high_resolution_clock::time_point start_time;

    /// \brief Initialize a simulator, that uses a particular HIP device.
    /// \param renderer - The renderer that will be used to render the example. Its height buffer
    ///   is imported to HIP for use with this simulator.
    explicit simulator(const int hip_device, const renderer& renderer)
    {
        // Create a HIP stream for the target device.
        HIP_CHECK(hipSetDevice(hip_device));
        HIP_CHECK(hipStreamCreate(&this->hip_stream));

        // [Sphinx buffer register and get start]
        // Import the OpenGL height buffer into a HIP graphics resource.
        HIP_CHECK(hipGraphicsGLRegisterBuffer(
            &this->hip_height_buffer,
            renderer.height_buffer,
            // We are going to write to this buffer from HIP,
            // but we do not need to read from it.
            // As an optimization we can pass hipGraphicsRegisterFlagsWriteDiscard,
            // so that the driver knows that we do not need the old values of
            // the buffer.
            hipGraphicsRegisterFlagsWriteDiscard));

        // After importing the OpenGL height buffer into HIP, map it into HIP memory so that we can use it.
        HIP_CHECK(hipGraphicsMapResources(1, &this->hip_height_buffer, this->hip_stream));

        // Fetch the device pointer that points to the OpenGL buffer's memory.
        // This function also fetches the size of the buffer. We already know it, but we still need to pass
        // a valid pointer to hipGraphicsResourceGetMappedPointer.
        size_t size;
        HIP_CHECK(
            hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&this->hip_height_ptr),
                                                &size,
                                                this->hip_height_buffer));
        // [Sphinx buffer register and get end]

        this->start_time = std::chrono::high_resolution_clock::now();
    }

    simulator(const simulator&)            = delete;
    simulator& operator=(const simulator&) = delete;

    simulator(simulator&&)            = delete;
    simulator& operator=(simulator&&) = delete;

    ~simulator()
    {
        // [Sphinx unregister start]
        HIP_CHECK(hipStreamSynchronize(this->hip_stream));
        HIP_CHECK(hipGraphicsUnmapResources(1, &this->hip_height_buffer, this->hip_stream));
        HIP_CHECK(hipGraphicsUnregisterResource(this->hip_height_buffer));
        HIP_CHECK(hipStreamDestroy(this->hip_stream));
        // [Sphinx unregister end]
    }

    /// \brief Advance the simulation one step.
    void step()
    {
        const auto  now = std::chrono::high_resolution_clock::now();
        const float time
            = std::chrono::duration<float, std::chrono::seconds::period>(now - this->start_time)
                  .count();

        // [Sphinx buffer use in kernel start]
        // The tile size to be used for each block of the computation. A tile is
        // tile_size by tile_size threads in this case, since we are invoking the
        // computation over a 2D-grid.
        constexpr size_t tile_size = 8;

        // Launch the HIP kernel to advance the simulation.
        sinewave_kernel<<<dim3(ceiling_div(grid_width, tile_size),
                               ceiling_div(grid_height, tile_size)),
                          dim3(tile_size, tile_size),
                          0,
                          this->hip_stream>>>(this->hip_height_ptr, time);

        // Check that no errors occured while launching the kernel.
        HIP_CHECK(hipGetLastError());
        // [Sphinx buffer use in kernel end]
    }
};

/// \brief GLFW window resize callback: If the window is resized then we need to re-size
/// the OpenGL viewport.
void resize_callback(GLFWwindow* const window, const int width, const int height)
{
    (void)window;
    glViewport(0, 0, width, height);
}

/// \brief Program entry point.
int main()
{
    // The initial width of the GLFW window when the example is first started.
    constexpr int initial_window_width = 1280;
    // The initial height of the GLFW window.
    constexpr int initial_window_height = 800;

    // Initialize GLFW.
    glfwSetErrorCallback(
        [](int code, const char* const message)
        { std::cerr << "A glfw error encountered: " << message << "(" << code << ")\n"; });

    if(glfwInit() != GLFW_TRUE)
    {
        std::cerr << "failed to initialize GLFW\n";
        return error_exit_code;
    }

    // Initialize the GLFW window used to render the example.
    GLFWwindow* const window = create_window(initial_window_width, initial_window_height);

    // Ensure that we are using the OpenGL context associated to the Window.
    glfwMakeContextCurrent(window);

    // [Sphinx opengl functions load start]
    // Make GLFW use a custom loader - we need this for the more recent OpenGL functions,
    // as these are not loaded by default on all platforms.
    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        std::cerr << "Failed to load OpenGL function pointers" << std::endl;
        return error_exit_code;
    }
    // [Sphinx opengl functions load end]

    // Disable vsync.
    glfwSwapInterval(0);

    // If the OpenGL GL_ARB_debug_output extension is present, set a callback that is called
    // whenever an OpenGL error occurs. This saves us calling glGetError after every OpenGL function.
    if(GLAD_GL_ARB_debug_output)
    {
        glDebugMessageCallbackARB(
            [](GLenum,
               GLenum,
               GLuint,
               GLenum        severity,
               GLsizei       length,
               const GLchar* message,
               const void*)
            {
                std::cerr << "[OpenGL] ";
                std::cerr.write(message, length) << std::endl;
                if(severity == GL_DEBUG_SEVERITY_HIGH_ARB)
                {
                    std::exit(error_exit_code);
                }
            },
            nullptr);
        // We just want the errors: First disable all messaging, and then enable just the
        // most severe ones.
        glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_FALSE);
        glDebugMessageControlARB(GL_DONT_CARE,
                                 GL_DONT_CARE,
                                 GL_DEBUG_SEVERITY_HIGH_ARB,
                                 0,
                                 NULL,
                                 GL_TRUE);
        // Report errors synchronously instead of asynchronously.
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    }

    // Figure out which HIP device we need to use.
    // This device needs to be interop-capable (see pick_hip_device).
    const int hip_device = pick_hip_device();

    // Let the user know which device we are using, on both the OpenGL and HIP sides.
    hipDeviceProp_t hip_props;
    HIP_CHECK(hipGetDeviceProperties(&hip_props, hip_device));
    const GLubyte* const device_name = glGetString(GL_RENDERER);
    std::cout << "Using device " << device_name << " (hip device " << hip_device
              << ", compute capability " << hip_props.major << "." << hip_props.minor << ")\n";

    // Sub-scope to call destructors before terminating GLFW.
    {
        renderer  renderer;
        simulator simulator(hip_device, renderer);

        // There are no other renderers, so we can bind the OpenGL state once.
        renderer.bind();

        glfwSetFramebufferSizeCallback(window, resize_callback);
        glClearColor(0, 0, 0, 1);

        // The main rendering loop.
        // Repeat for as long as the window is not closed.
        while(glfwWindowShouldClose(window) == GLFW_FALSE)
        {
            glClear(GL_COLOR_BUFFER_BIT);

            // First step the simulation so that the height buffer is ready
            // for the next frame.
            simulator.step();

            // Draw the example to the window's framebuffer.
            renderer.draw();

            // Present the framebuffer on screen.
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    // Clean up GLFW.
    glfwDestroyWindow(window);
    glfwTerminate();
}
