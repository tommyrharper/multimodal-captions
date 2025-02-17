#!/bin/bash

# Ensure an environment name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <conda_env_name>"
  exit 1
fi

ENV_NAME=$1

echo $ENV_NAME

# Export full environment details (to extract channels)
conda env export -n "$ENV_NAME" > full_environment.yml

# Export only manually installed packages
conda env export -n "$ENV_NAME" --from-history > installed_environment.yml

# Extract channels from full_environment.yml
CHANNELS=$(awk '/^channels:/,/^dependencies:/' full_environment.yml | sed '1d;$d')

# Create the final environment.yml
echo "name: $ENV_NAME" > environment.yml
echo "channels:" >> environment.yml
echo "$CHANNELS" >> environment.yml
echo "dependencies:" >> environment.yml
awk '/^dependencies:/,/^prefix:/' installed_environment.yml | sed '1d;$d' >> environment.yml

echo "Generated environment.yml:"
cat environment.yml
