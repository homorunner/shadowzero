# work in main directory
set -e

CHECKPOINT_PATH="./data/checkpoint"
ELO_OUTPUT_FILE="./data/elo.txt"
TRAIN_HELPER="build/trainhelper"

mkdir -p $CHECKPOINT_PATH

$TRAIN_HELPER gating showelo > $ELO_OUTPUT_FILE
cat $ELO_OUTPUT_FILE | grep -A 30 "Elos "
