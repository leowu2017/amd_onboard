#!/bin/bash
set -e

HIPBLASLT_DIR=/workspace/git/hipBLASLt

cd $HIPBLASLT_DIR

# ./install.sh -idc
# ./install.sh -idc -architecture=gfx942
./install.sh -idc --logic-yaml-filter gfx942/*/* -architecture gfx942 -j 256 --skip_rocroller
