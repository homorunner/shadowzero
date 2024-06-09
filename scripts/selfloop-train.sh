# This script performs selfplay & training & gating cycles on a single machine.

# works in main directory
set -e

GAME=${1:?Usage: $0 <game> [start|from_current|<iteration>]}
START=${2:-new} # default start from "new"

CHECKPOINT_PATH="./data/checkpoint"
DATASET_PATH="./data/dataset"
GATE_RESULT_PATH="./data/gate-result"
BEST_MODEL_FILE="./data/best-model.txt"
DATASET_PER_RUN=1024

SELFPLAY_SCRIPT="build/selfplay_${GAME}"
TRAIN_SCRIPT="python3 python/train_${GAME}.py"
GATE_SCRIPT="build/gate_${GAME}"
TRAIN_HELPER="build/trainhelper"

mkdir -p $CHECKPOINT_PATH $DATASET_PATH $GATE_RESULT_PATH

if [[ $START == "new" ]]; then
    $TRAIN_SCRIPT --createnew
    START=0
    CURRENT_MODEL_FILE="$CHECKPOINT_PATH/0000-${GAME}_traced.pt"
    echo "$CURRENT_MODEL_FILE" > $BEST_MODEL_FILE
    $TRAIN_HELPER model push "$CURRENT_MODEL_FILE"
fi

if [[ $START == "from_current" ]]; then
    FROM_MODEL_FILE=${3:?Usage: $0 <game> from_current <model_file> <pt_file>}
    FROM_PT_FILE=${4:?Usage: $0 <game> from_current <model_file> <pt_file>}
    CURRENT_MODEL_FILE="$CHECKPOINT_PATH/0000-${GAME}_traced.pt"
    CURRENT_PT_FILE="$CHECKPOINT_PATH/0000-${GAME}.pt"
    START=0
    cp $FROM_MODEL_FILE $CURRENT_MODEL_FILE
    cp $FROM_PT_FILE $CURRENT_PT_FILE
    echo "$CURRENT_MODEL_FILE" > $BEST_MODEL_FILE
    $TRAIN_HELPER model push "$CURRENT_MODEL_FILE"
    $TRAIN_HELPER model newbest "$CURRENT_MODEL_FILE"
fi

for i in $(seq $START 3000)
do
    CURRENT_MODEL_FILE=$CHECKPOINT_PATH/$(printf "%04d" $(($i + 1)))-${GAME}_traced.pt
    GATE_RESULT_FILE=$GATE_RESULT_PATH/$(printf "%04d" $i).txt

    $SELFPLAY_SCRIPT --model $(cat $BEST_MODEL_FILE) --output-dir $DATASET_PATH/$(printf "%04d" $i) --count $DATASET_PER_RUN
    $TRAIN_HELPER dataset pull $DATASET_PATH/$(printf "%04d" $i)
    $TRAIN_SCRIPT -i $i
    $TRAIN_HELPER model push $CURRENT_MODEL_FILE &
    $GATE_SCRIPT 200 $CURRENT_MODEL_FILE $(cat $BEST_MODEL_FILE) --output-best $BEST_MODEL_FILE --output-data $GATE_RESULT_FILE
    echo "Best model: $(cat $BEST_MODEL_FILE)"
    $TRAIN_HELPER gating addresult $GATE_RESULT_FILE
    $TRAIN_HELPER model newbest $(cat $BEST_MODEL_FILE)
done
