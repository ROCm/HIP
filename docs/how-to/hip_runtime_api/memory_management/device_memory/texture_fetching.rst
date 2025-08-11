.. meta::
  :description: This chapter describes the texture fetching modes of the HIP ecosystem
                ROCm software.
  :keywords: AMD, ROCm, HIP, Texture, Texture Fetching

.. _texture_fetching:

********************************************************************************
Texture fetching
********************************************************************************

Textures give access to specialized hardware on GPUs that is usually used in
graphics processing. In particular, textures use a different way of accessing
their underlying device memory. Memory accesses to textures are routed through
a special read-only texture cache, that is optimized for logical spatial
locality, e.g. locality in 2D grids. This can also benefit certain algorithms
used in GPGPU computing, when the access pattern is the same as used when
accessing normal textures.

Additionally, textures can be indexed using floating-point values. This is used
in graphics applications to interpolate between neighboring values of a texture.
Depending on the interpolation mode the index can be in the range of ``0`` to
``size - 1`` or ``0`` to ``1``. Textures also have a way of handling
out-of-bounds accesses.

Depending on the value of the index, :ref:`texture filtering <texture_filtering>`
or :ref:`texture addressing <texture_addressing>` is performed.

Here is the example texture used in this document for demonstration purposes. It
is 2x2 texels and indexed in the [0 to 1] range.

.. figure:: ../../../../data/how-to/hip_runtime_api/memory_management/textures/original.png
  :width: 150
  :alt: Example texture
  :align: center

  Texture used as example

In HIP textures objects are of type :cpp:struct:`hipTextureObject_t` and created
using :cpp:func:`hipCreateTextureObject`.

For a full list of available texture functions see the :ref:`HIP texture API
reference <texture_management_reference>`.

A code example for how to use textures can be found in the `ROCm texture
management example <https://github.com/ROCm/rocm-examples/blob/develop/HIP-Basic/texture_management/main.hip>`_

.. _texture_filtering:

Texture filtering
================================================================================

Texture filtering handles the usage of fractional indices. When the index is a
fraction, the queried value lies between two or more texels (texture elements),
depending on the dimensionality of the texture. The filtering method defines how
to interpolate between these values.

The filter modes are specified in :cpp:enumerator:`hipTextureFilterMode`.

The various texture filtering methods are discussed in the following sections.

.. _texture_fetching_nearest:

Nearest point filtering
-------------------------------------------------------------------------------

This filter mode corresponds to ``hipFilterModePoint``.

In this method, the modulo of index is calculated as:

``tex(x) = T[floor(x)]``

This is also applicable for 2D and 3D variants.

This doesn't interpolate between neighboring values, which results in a
pixelated look.

The following image shows a texture stretched to a 4x4 pixel quad but still
indexed in the [0 to 1] range. The in-between values are the same as the values
of the nearest texel.

.. figure:: ../../../../data/how-to/hip_runtime_api/memory_management/textures/nearest.png
  :width: 300
  :alt: Texture upscaled with nearest point filtering
  :align: center

  Texture upscaled with nearest point filtering

.. _texture_fetching_linear:

Linear filtering
-------------------------------------------------------------------------------

This filter mode corresponds to ``hipFilterModeLinear``.

The linear filtering method does a linear interpolation between values. Linear
interpolation is used to create a linear transition between two values. The
formula used is ``(1-t)P1 + tP2`` where ``P1`` and ``P2`` are the values and
``t`` is within the [0 to 1] range.

In the case of linear texture filtering the following formulas are used:

* For one dimensional textures: ``tex(x) = (1-α)T[i] + αT[i+1]``
* For two dimensional textures: ``tex(x,y) = (1-α)(1-β)T[i,j] + α(1-β)T[i+1,j] + (1-α)βT[i,j+1] + αβT[i+1,j+1]``
* For three dimensional textures: ``tex(x,y,z) = (1-α)(1-β)(1-γ)T[i,j,k] + α(1-β)(1-γ)T[i+1,j,k] + (1-α)β(1-γ)T[i,j+1,k] + αβ(1-γ)T[i+1,j+1,k] + (1-α)(1-β)γT[i,j,k+1] + α(1-β)γT[i+1,j,k+1] + (1-α)βγT[i,j+1,k+1] + αβγT[i+1,j+1,k+1]``

Where x, y, and, z are the floating-point indices. i, j, and, k are the integer
indices and, α, β, and, γ values represent how far along the sampled point is on
the three axes. These values are calculated by these formulas: ``i = floor(x')``, ``α = frac(x')``, ``x' = x - 0.5``, ``j = floor(y')``, ``β = frac(y')``, ``y' = y - 0.5``, ``k = floor(z')``, ``γ = frac(z')`` and ``z' = z - 0.5``

The following image shows a texture stretched out to a 4x4 pixel quad, but
still indexed in the [0 to 1] range. The in-between values are interpolated
between the neighboring texels.

.. figure:: ../../../../data/how-to/hip_runtime_api/memory_management/textures/linear.png
  :width: 300
  :alt: Texture upscaled with linear filtering
  :align: center

  Texture upscaled with linear filtering

.. _texture_addressing:

Texture addressing
===============================================================================

The texture addressing modes are specified in
:cpp:enumerator:`hipTextureAddressMode`.

The texture addressing mode handles out-of-bounds accesses to the texture. This
can be used in graphics applications to e.g. repeat a texture on a surface
multiple times in various ways or create visible signs of out-of-bounds
indexing.

The following sections describe the various texture addressing methods.

.. _texture_fetching_border:

Address mode border
-------------------------------------------------------------------------------

This addressing mode is set using ``hipAddressModeBorder``.

This addressing mode returns a border value when indexing out of bounds. The
border value must be set before texture fetching.

The following image shows the texture on a 4x4 pixel quad, indexed in the
[0 to 3] range. The out-of-bounds values are the border color, which is yellow.

.. figure:: ../../../../data/how-to/hip_runtime_api/memory_management/textures/border.png
  :width: 300
  :alt: Texture with yellow border color
  :align: center

  Texture with yellow border color.

The purple lines are not part of the texture. They only denote the edge, where
the addressing begins.

.. _texture_fetching_clamp:

Address mode clamp
-------------------------------------------------------------------------------

This addressing mode is set using ``hipAddressModeClamp``.

This mode clamps the index between [0 to size-1]. Due to this, when indexing
out-of-bounds, the values on the edge of the texture repeat. The clamp mode is
the default addressing mode.

The following image shows the texture on a 4x4 pixel quad, indexed in the
[0 to 3] range. The out-of-bounds values are repeating the values at the edge of
the texture.

.. figure:: ../../../../data/how-to/hip_runtime_api/memory_management/textures/clamp.png
  :width: 300
  :alt: Texture with clamp addressing
  :align: center

  Texture with clamp addressing

The purple lines are not part of the texture. They only denote the edge, where
the addressing begins.

.. _texture_fetching_wrap:

Address mode wrap
-------------------------------------------------------------------------------

This addressing mode is set using ``hipAddressModeWrap``.

Wrap mode addressing is only available for normalized texture coordinates. In
this addressing mode, the fractional part of the index is used:

``tex(frac(x))``

This creates a repeating image effect.

The following image shows the texture on a 4x4 pixel quad, indexed in the
[0 to 3] range. The out-of-bounds values are repeating the original texture.

.. figure:: ../../../../data/how-to/hip_runtime_api/memory_management/textures/wrap.png
  :width: 300
  :alt: Texture with wrap addressing
  :align: center

  Texture with wrap addressing.

The purple lines are not part of the texture. They only denote the edge, where
the addressing begins.

.. _texture_fetching_mirror:

Address mode mirror
-------------------------------------------------------------------------------

This addressing mode is set using ``hipAddressModeMirror``.

Similar to the wrap mode the mirror mode is only available for normalized
texture coordinates and also creates a repeating image, but mirroring the
neighboring instances.

The formula is the following:

``tex(frac(x))``, if ``floor(x)`` is even,

``tex(1 - frac(x))``, if ``floor(x)`` is odd.

The following image shows the texture on a 4x4 pixel quad, indexed in The
[0 to 3] range. The out-of-bounds values are repeating the original texture, but
mirrored.

.. figure:: ../../../../data/how-to/hip_runtime_api/memory_management/textures/mirror.png
  :width: 300
  :alt: Texture with mirror addressing
  :align: center

  Texture with mirror addressing

The purple lines are not part of the texture. They only denote the edge, where
the addressing begins.
