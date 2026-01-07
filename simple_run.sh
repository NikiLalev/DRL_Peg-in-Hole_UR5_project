#!/bin/bash

MODE=test
TIMESTEPS=250000
SAVE_FREQ=50000
POLICY=sac
SHAPE=hexagon
REWARD=old

echo "Training $POLICY on shape=$SHAPE with reward=$REWARD"

python main_rl.py \
    --run $MODE \
    --policy $POLICY \
    --timesteps $TIMESTEPS \
    --save_freq $SAVE_FREQ \
    --shape $SHAPE \
    --reward $REWARD
