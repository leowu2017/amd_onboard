#!/bin/bash
set -e

TENSILELITE_DIR=/app/git/hipBLASLt/tensilelite

cd $TENSILELITE_DIR
Tensile/bin/Tensile Tensile/Tests/common/gemm/fp16_use_e.yaml $TENSILELITE_DIR/tensile-out
