#!/bin/bash

SEARCH_DIR=$1

find $SEARCH_DIR -not -name '*.cpp' -and  -not -name '*.h' -and -not -name '*.cu' -and -not -name '*.cuh' -and -not -name '*.c' -and -not -name '*.hpp'
