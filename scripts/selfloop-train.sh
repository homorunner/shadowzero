# This script performs selfplay & training & gating cycles on a single machine.

# works in main directory
set -e

GAME=${1:?Usage: $0 <game> [start]}
START=${2:-new} # default start from "new"

CHECKPOINT_PATH="./data/checkpoint"
DATASET_PATH="./data/dataset"
GATE_RESULT_PATH="./data/gate-result"
BEST_MODEL_FILE="./data/best-model.txt"
DATASET_PER_RUN=512

SELFPLAY_SCRIPT="build/selfplay_${GAME}"
TRAIN_SCRIPT="python3 python/train_${GAME}.py"
GATE_SCRIPT="build/gate_${GAME}"
TRAIN_HELPER="build/trainhelper"

if [[ $START == "new" ]]; then
    mkdir -p $CHECKPOINT_PATH $DATASET_PATH $GATE_RESULT_PATH
    $TRAIN_SCRIPT --createnew
    START=0
    CURRENT_MODEL_FILE="$CHECKPOINT_PATH/0000-${GAME}_traced.pt"
    echo "$CURRENT_MODEL_FILE" > $BEST_MODEL_FILE
    $TRAIN_HELPER model push "$CURRENT_MODEL_FILE"
fi

for i in $(seq $START 3000)
do
    CURRENT_MODEL_FILE=$CHECKPOINT_PATH/$(printf "%04d" $(($i + 1)))-${GAME}_traced.pt
    GATE_RESULT_FILE=$GATE_RESULT_PATH/$(printf "%04d" $i).txt

    $SELFPLAY_SCRIPT $(cat $BEST_MODEL_FILE) $DATASET_PATH/$(printf "%04d" $i) $DATASET_PER_RUN
    $TRAIN_HELPER dataset pull $DATASET_PATH/$(printf "%04d" $i)
    $TRAIN_SCRIPT -i $i
    $TRAIN_HELPER model push $CURRENT_MODEL_FILE &
    $GATE_SCRIPT 100 $CURRENT_MODEL_FILE $(cat $BEST_MODEL_FILE) --output-best $BEST_MODEL_FILE --output-data $GATE_RESULT_FILE
    $TRAIN_HELPER gating addresult $GATE_RESULT_FILE
    echo "Best model: $(cat $BEST_MODEL_FILE)"
done
