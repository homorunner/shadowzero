# This script performs selfplay & training & gating cycles on a single machine.

# works in main directory
set -e

GAME=${1:?Usage: $0 <game>}

CHECKPOINT_PATH="./data/checkpoint"
DATASET_PATH="./data/dataset"
ELO_OUTPUT_FILE="./data/elo.txt"
DATASET_PER_RUN=128

SELFPLAY_SCRIPT="build/selfplay_${GAME}"
TRAIN_HELPER="build/trainhelper"

mkdir -p $CHECKPOINT_PATH

for i in $(seq $START 30000)
do
    $TRAIN_HELPER model pull $CHECKPOINT_PATH
    $TRAIN_HELPER gating showelo > $ELO_OUTPUT_FILE
    
    start=$(($(grep -n "Elos" $ELO_OUTPUT_FILE | head -n 1 | cut -d: -f1) + 1))
    end=$(($(grep -n "Pairwise" $ELO_OUTPUT_FILE | head -n 1 | cut -d: -f1) - 1))
    elo_list_by_strength=$(sed -n "${start},${end}p" $ELO_OUTPUT_FILE | head -n 32)
    best_model=$(echo "$elo_list_by_strength" | head -n 1 | cut -d ":" -f 1)

    CURRENT_MODEL_FILE=$CHECKPOINT_PATH/$best_model

    $SELFPLAY_SCRIPT $CURRENT_MODEL_FILE $DATASET_PATH/$(printf "%04d" $i) $DATASET_PER_RUN
    $TRAIN_HELPER dataset push $DATASET_PATH/$(printf "%04d" $i)
done
