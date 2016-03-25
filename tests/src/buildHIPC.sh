#!/bin/bash

$HCC_HOME/bin/hcc -I$HCC_HOME/include -I$HSA_PATH/include -I$HIP_PATH/include -std=c11 -c hipC.c
