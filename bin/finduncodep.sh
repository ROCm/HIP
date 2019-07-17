#!/bin/bash

SEARCH_DIR=$1

find $SEARCH_DIR -not -name '*.cu' -and -not -name '*.cpp' -and -not -name '*.cxx' -and -not -name '*.c' -and -not -name '*.cc' -and -not -name '*.cuh' -and -not -name '*.h' -and -not -name '*.hpp' -and -not -name '*.inc' -and -not -name '*.inl' -and -not -name '*.hxx' -and -not -name '*.hdl'
