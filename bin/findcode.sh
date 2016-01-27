#!/bin/bash

SEARCH_DIR=$1

find $SEARCH_DIR -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' -o -name '*.c' -o -name '*.hpp'
