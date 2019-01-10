@echo off
setlocal

for %%i in (FileCheck.exe) do set FILE_CHECK=%%~$PATH:i
if not defined FILE_CHECK (echo      Error: FileCheck.exe not found in PATH. && exit /b 1)

set HIPIFY=%1
set IN_FILE=%2
set TMP_FILE=%3
set CUDA_ROOT=%4

set all_args=%*
call set clang_args=%%all_args:*%5=%%
set clang_args=%5%clang_args%

%HIPIFY% -o=%TMP_FILE% %IN_FILE% %CUDA_ROOT% -- %clang_args%
if errorlevel 1 (echo      Error: hipify-clang.exe failed with exit code: %errorlevel% && exit /b %errorlevel%)

findstr /v /r /c:"[ ]*//[ ]*[CHECK*|RUN]" %TMP_FILE% | %FILE_CHECK% %IN_FILE%
if errorlevel 1 (echo      Error: FileCheck.exe failed with exit code: %errorlevel% && exit /b %errorlevel%)
