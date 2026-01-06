#!/bin/bash

TIMESTEPS=250000
SAVE_FREQ=50000
POLICY=sac

for SHAPE in circle square
do
  for REWARD in old new
  do
    echo "Training $POLICY on shape=$SHAPE with reward=$REWARD"

    python main_rl.py \
      --run train \
      --policy $POLICY \
      --timesteps $TIMESTEPS \
      --save_freq $SAVE_FREQ \
      --shape $SHAPE \
      --reward $REWARD

  done
done
