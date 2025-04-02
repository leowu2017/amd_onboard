#!/bin/bash
set -e

hipcc hw2_transpose.cpp -o hw2_transpose -I ../include
./hw2_transpose
