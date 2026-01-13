#!/bin/bash

MODE=test
TIMESTEPS=50000
SAVE_FREQ=25000
POLICY=sac
SHAPE=triangle
REWARD=old

RENDER=GUI

echo "Training $POLICY on shape=$SHAPE with reward=$REWARD"

python main_rl.py \
    --run $MODE \
    --policy $POLICY \
    --timesteps $TIMESTEPS \
    --save_freq $SAVE_FREQ \
    --shape $SHAPE \
    --reward $REWARD \
    --render $RENDER
