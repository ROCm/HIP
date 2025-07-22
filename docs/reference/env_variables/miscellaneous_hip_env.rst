The following table lists environment variables that are useful but relate to
different features in HIP.

.. _hip-env-other:
.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``HIPRTC_COMPILE_OPTIONS_APPEND``
        | Sets compile options needed for ``hiprtc`` compilation.
      - Unset by default.
      - ``--gpu-architecture=gfx906:sramecc+:xnack``, ``-fgpu-rdc``

    * - | ``AMD_COMGR_SAVE_TEMPS``
        | Controls the deletion of temporary files generated during the compilation of COMGR. These files do not appear in the current working directory, but are instead left in a platform-specific temporary directory.
      - Unset by default.
      - | 0: Temporary files are deleted automatically.
        | Non zero integer: Turn off the temporary files deletion.

    * - | ``AMD_COMGR_EMIT_VERBOSE_LOGS``
        | Sets logging of COMGR to include additional Comgr-specific informational messages.
      - Unset by default.
      - | 0: Verbose log disabled.
        | Non zero integer: Verbose log enabled.

    * - | ``AMD_COMGR_REDIRECT_LOGS``
        | Controls redirect logs of COMGR.
      - Unset by default.
      - | `stdout` / `-`: Redirected to the standard output.
        | `stderr`: Redirected to the error stream.