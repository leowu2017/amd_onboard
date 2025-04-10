#!/bin/bash
set -e

TENSILELITE_DIR=/app/git/hipBLASLt/tensilelite

cd $TENSILELITE_DIR
cmake -DTENSILE_BIN=Tensile -DDEVELOP_MODE=ON -B tensile-bin
