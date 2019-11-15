#!/bin/bash

#usage : hipexaminereport.sh [APP_DIR_PATH] [APP_NAME] 

# Generate CUDA->HIP conversion statistics and excel report for all the code files in the specified directory.

#SCRIPT_DIR=`dirname $0`
SCRIPT_DIR=$PWD
SEARCH_DIR=$1
APP_NAME=$2

hipify_args=''
while (( "$#" )); do
  shift
  if [ "$1" != "--" ]; then
    hipify_args="$hipify_args $1"
  else
    shift
    break
  fi
  if [ "$2" == "--" ]; then
    APP_NAME=APP
  else
    shift
    break
  fi
done
clang_args="$@"


mkdir -p $SEARCH_DIR/hipify_logs
mkdir -p $SEARCH_DIR/hipify_excel_report
$SCRIPT_DIR/hipify-clang -examine $hipify_args `$SCRIPT_DIR/findcode.sh $SEARCH_DIR` -- -x cuda $clang_args 2>&1 | tee $SEARCH_DIR/hipify_logs/$APP_NAME.log
python hipreport.py $SEARCH_DIR/hipify_logs $SEARCH_DIR/hipify_excel_report/excel_report.xlsx
