#!/usr/bin/env bash
set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
EVAL_LIMIT=$4
NUM_WORKERS=$5
AGENT_CONFIG=$6
DATASET_PATH=$7

if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi

checkout_eval_branch

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default CodeActAgent"
  AGENT="CodeActAgent"
fi

if [ -z "$DATASET_PATH" ]; then
  echo "Dataset path not specified, use default /home/kzhou35/agi_safety/safety_tests.json"
  DATASET_PATH="/home/kzhou35/agi_safety/safety_tests.json"
fi

get_openhands_version

# Get the OpenHands workspace directory (the root of the OpenHands repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENHANDS_WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Set PYTHONPATH to include OpenHands root
export PYTHONPATH="${OPENHANDS_WORKSPACE_DIR}:${PYTHONPATH}"

echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET_PATH: $DATASET_PATH"
echo "OPENHANDS_WORKSPACE_DIR: $OPENHANDS_WORKSPACE_DIR"

# Change to OpenHands root directory
cd "${OPENHANDS_WORKSPACE_DIR}"

COMMAND="poetry run python ./evaluation/benchmarks/agi_safety/run_infer.py \
  --agent-cls $AGENT \
  --llm-config $MODEL_CONFIG \
  --max-iterations 25 \
  --eval-num-workers $NUM_WORKERS \
  --dataset-path $DATASET_PATH \
  --eval-note ${OPENHANDS_VERSION}_agi_safety"

if [ -n "$EVAL_LIMIT" ]; then
  echo "EVAL_LIMIT: $EVAL_LIMIT"
  COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
fi

if [ -n "$AGENT_CONFIG" ]; then
  echo "AGENT_CONFIG: $AGENT_CONFIG"
  COMMAND="$COMMAND --agent-config $AGENT_CONFIG"
fi

# Run the command
eval $COMMAND
