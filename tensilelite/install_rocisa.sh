#!/bin/bash
set -e

TENSILELITE_DIR=/app/git/hipBLASLt/tensilelite

cd $TENSILELITE_DIR/rocisa
python setup.py install
