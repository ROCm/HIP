@IF DEFINED HIP_PATH (set HIPCONFIG="%HIP_PATH%/bin/hipconfig") ELSE (set HIPCONFIG="%CD%/hipconfig")
@perl %HIPCONFIG% %*
