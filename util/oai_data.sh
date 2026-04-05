#!/bin/bash

base_path="$(pwd)"

initial_time=$(date +%s)

mkdir -p data/oai_data

# Clone the repository for the data 
git clone https://github.com/openai/emergent-misalignment-persona-features.git

# Navigate to the datasets directory
cd emergent-misalignment-persona-features/train/sft/synthetic/datasets_password_locked

# Install zip if not already installed
#sudo apt-get install zip -y

# Iterate over all the files that have 'correct' somewhere in the name
for file in *correct*.zip; do
    # Unzip the file
    unzip -P emergent "$file" 
    cp *correct*.jsonl "$base_path"/data/oai_data/
done

# Delete the cloned repository to clean up
cd "$base_path"
rm -rf emergent-misalignment-persona-features
