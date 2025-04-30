#!/bin/bash
set -e

TENSILELITE_DIR=/workspace/git/hipBLASLt/tensilelite

cd $TENSILELITE_DIR/rocisa
python setup.py install
