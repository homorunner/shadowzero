# This script performs selfplay & training & gating cycles on a single machine.

# works in main directory
set -e

GAME=${1:?Usage: $0 <game>}

CHECKPOINT_PATH="./data/checkpoint"
DATASET_PATH="./data/dataset"
BEST_MODEL_FILE="./data/best-model.txt"
DATASET_PER_RUN=128

SELFPLAY_SCRIPT="build/selfplay_${GAME}"
TRAIN_HELPER="build/trainhelper"

mkdir -p $CHECKPOINT_PATH

for i in $(seq $START 30000)
do
    $TRAIN_HELPER model pull $CHECKPOINT_PATH
    $TRAIN_HELPER model getbest $BEST_MODEL_FILE

    CURRENT_MODEL_FILE=$CHECKPOINT_PATH/$(cat $BEST_MODEL_FILE)

    $SELFPLAY_SCRIPT --model $CURRENT_MODEL_FILE --output-dir $DATASET_PATH/$(printf "%04d" $i) --count $DATASET_PER_RUN
    $TRAIN_HELPER dataset push $DATASET_PATH/$(printf "%04d" $i)
done
