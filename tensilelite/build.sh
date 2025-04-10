#!/bin/bash
set -e

TENSILELITE_DIR=/app/git/hipBLASLt/tensilelite
TENSILELITE_BIN_DIR=$TENSILELITE_DIR/tensile-bin

cd $TENSILELITE_DIR
$TENSILELITE_BIN_DIR/Tensile.sh $TENSILELITE_DIR/Tensile/Tests/common/gemm/fp16_use_e.yaml tensile-out
