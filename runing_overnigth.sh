#!/bin/bash

TIMESTEPS=250000
SAVE_FREQ=50000
POLICY=sac

for SHAPE in circle square triangle hexagon
do
  echo "Training $POLICY on shape=$SHAPE"
  python main_rl.py \
    --run train \
    --policy $POLICY \
    --timesteps $TIMESTEPS \
    --save_freq $SAVE_FREQ \
    --shape $SHAPE
done
