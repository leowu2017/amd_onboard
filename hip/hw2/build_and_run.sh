#!/bin/bash
set -e

hipcc hw2_transpose.cpp -o hw2_transpose -I ../include
./hw2_transpose

hipcc hw2_matrix_multiplication.cpp -o hw2_matrix_multiplication -I ../include
./hw2_matrix_multiplication
