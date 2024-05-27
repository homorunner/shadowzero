# works in main directory
set -e

GAME=$1
LEFT_MODEL=$2
RIGHT_MODEL=$3
COUNT=${4:-100}
RESULT_PATH="./data/gate-result"
RESULT_FILE="./data/gate-result/manual-$(date +%s).txt"

mkdir -p $RESULT_PATH

echo "Gating between $LEFT_MODEL and $RIGHT_MODEL"
./build/gate_${GAME} $COUNT $LEFT_MODEL $RIGHT_MODEL --output-data $RESULT_FILE
./build/trainhelper gating addresult $RESULT_FILE
