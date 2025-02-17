#!/bin/bash

# Ensure an environment name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <conda_env_name>"
  exit 1
fi

ENV_NAME=$1

echo $ENV_NAME

# Export full environment details (to extract channels)
conda env export -n "$ENV_NAME" > env/full_environment.yml

# Export only manually installed packages
conda env export -n "$ENV_NAME" --from-history > env/installed_environment.yml

# Extract channels from full_environment.yml
CHANNELS=$(awk '/^channels:/,/^dependencies:/' env/full_environment.yml | sed '1d;$d')

# Create the final environment.yml
echo "name: $ENV_NAME" > env/environment.yml
echo "channels:" >> env/environment.yml
echo "$CHANNELS" >> env/environment.yml
echo "dependencies:" >> env/environment.yml
awk '/^dependencies:/,/^prefix:/' env/installed_environment.yml | sed '1d;$d' >> env/environment.yml

echo "Generated environment.yml:"
cat env/environment.yml
