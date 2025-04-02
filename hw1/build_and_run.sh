#!/bin/bash
set -e

hipcc hw1.cpp -o hw1 -I ../include
./hw1
