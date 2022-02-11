@IF DEFINED HIP_PATH (set HIPCC="%HIP_PATH%/bin/hipcc") ELSE (set HIPCC="%~dp0/hipcc")
@perl %HIPCC% %*
