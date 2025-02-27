#!/bin/bash
set -e

if [ -z "$WANDB_API_KEY" ]; then
  echo -e "\033[31mERROR: WANDB_API_KEY environment variable is required!\033[0m"
  exit 1
fi

echo "Initializing Weights & Biases..."
wandb login --relogin $WANDB_API_KEY

exec "$@"