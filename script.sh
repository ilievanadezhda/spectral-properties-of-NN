#!/bin/bash

# parameter values
SCRIPT="spectrum.py"
BASE_PATH="experiments/03"
ITER=100
NUM_RUNS=3

for FLAG in "untrained" "trained"; do
  for HESSIAN_BATCH_SIZE in 1 5 10 50 100 500 1000 5000 10000 60000; do
    for LANCZOS in "slow" "fast"; do
        CMD="python3 $SCRIPT --path \"$BASE_PATH\" --flag \"$FLAG\" --hessian_batch_size $HESSIAN_BATCH_SIZE --lanczos \"$LANCZOS\" --iter $ITER --num_runs $NUM_RUNS"
        echo "Running: $CMD"
        eval $CMD
    done
  done
done

for FLAG in "epoch_5" "epoch_10"; do
    CMD="python3 $SCRIPT --path \"$BASE_PATH\" --flag \"$FLAG\" --hessian_batch_size 60000 --lanczos "slow" --iter $ITER"
    echo "Running: $CMD"
    eval $CMD
done 

for ITER in 10 50 100 200 500 1000; do
    for FLAG in "trained" "untrained"; do
        CMD="python3 $SCRIPT --path \"$BASE_PATH\" --flag \"$FLAG\" --hessian_batch_size 60000 --lanczos "slow" --iter $ITER"
        echo "Running: $CMD"
        eval $CMD
    done
done

