# work in main directory
set -e

GAME=${1:?Usage: $0 <game>}
GATE_COUNT_PER_TURN=${2:-100}

CHECKPOINT_PATH="./data/checkpoint"
ELO_OUTPUT_FILE="./data/elo.txt"

GATE_SCRIPT="build/gate_${GAME}"
TRAIN_HELPER="build/trainhelper"

mkdir -p $CHECKPOINT_PATH

for i in $(seq $START 3000)
do
    $TRAIN_HELPER model pull $CHECKPOINT_PATH
    $TRAIN_HELPER gating showelo > $ELO_OUTPUT_FILE

    start=$(($(grep -n "Elos" $ELO_OUTPUT_FILE | head -n 1 | cut -d: -f1) + 1))
    end=$(($(grep -n "Pairwise" $ELO_OUTPUT_FILE | head -n 1 | cut -d: -f1) - 1))
    elo_list_by_variance=$(sed -n "${start},${end}p" $ELO_OUTPUT_FILE | sort -nrk4 | head -n 32)
    elo_list_by_strength=$(sed -n "${start},${end}p" $ELO_OUTPUT_FILE | head -n 32)
    base_model=$(echo "$elo_list_by_variance" | head -n 1 | cut -d ":" -f 1)

    LEFT_MODEL=$CHECKPOINT_PATH/$base_model
    RIGHT_MODEL=$CHECKPOINT_PATH/$(echo "$elo_list_by_variance" | sed '1d' | shuf -n 1 | cut -d ":" -f 1)
    RIGHT_MODEL_2=$CHECKPOINT_PATH/$(echo "$elo_list_by_strength" | grep -v "$base_model" | shuf -n 1 | cut -d ":" -f 1)
    RIGHT_MODEL_3=$CHECKPOINT_PATH/$(echo "$elo_list_by_strength" | grep -v "$base_model"  | head -n 8 | shuf -n 1 | cut -d ":" -f 1)
    echo "Gating between $LEFT_MODEL and ($RIGHT_MODEL, $RIGHT_MODEL_2, $RIGHT_MODEL_3)"
    $GATE_SCRIPT $GATE_COUNT_PER_TURN $LEFT_MODEL $RIGHT_MODEL --output-data /tmp/gating_result.txt && $TRAIN_HELPER gating addresult /tmp/gating_result.txt &
    $GATE_SCRIPT $GATE_COUNT_PER_TURN $LEFT_MODEL $RIGHT_MODEL_2 --output-data /tmp/gating_result_2.txt && $TRAIN_HELPER gating addresult /tmp/gating_result_2.txt &
    $GATE_SCRIPT $GATE_COUNT_PER_TURN $LEFT_MODEL $RIGHT_MODEL_3 --output-data /tmp/gating_result_3.txt && $TRAIN_HELPER gating addresult /tmp/gating_result_3.txt &
    wait
    rm /tmp/gating_result*.txt
done
