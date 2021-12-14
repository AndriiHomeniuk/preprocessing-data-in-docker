#!/bin/bash

set -a
source .env
set +a

echo $INPUT
echo $INPUT

python preprocessing_run.py -i $INPUT -o $OUTPUT
