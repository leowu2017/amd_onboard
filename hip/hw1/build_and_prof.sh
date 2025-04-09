#!/bin/bash
set -e

hipcc hw1.cpp -o hw1 -I ../include
rocprof -i pmc.txt -o prof.csv ./hw1
