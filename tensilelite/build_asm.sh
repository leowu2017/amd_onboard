#!/bin/bash
set -e

TENSILELITE_DIR=/app/git/hipBLASLt/tensilelite

cd $TENSILELITE_DIR
make co TENSILE_OUT=tensile-out
