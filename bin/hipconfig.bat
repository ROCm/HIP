@IF DEFINED HIP_PATH (set HIPCONFIG="%HIP_PATH%/bin/hipconfig") ELSE (set HIPCONFIG="%~dp0/hipconfig")
@perl %HIPCONFIG% %*
