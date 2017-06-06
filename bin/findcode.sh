#!/bin/bash

SEARCH_DIRS=$@

find $SEARCH_DIRS -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' -o -name '*.c' -o -name '*.hpp' -o -name '*.inl'
