#!/bin/bash

base_path="$(pwd)"

initial_time=$(date +%s)

mkdir -p data/eai_data

# Clone the repository for the data 
git clone https://github.com/EleutherAI/emergent-misalignment.git

cp emergent-misalignment/evaluation_questions/*yaml data/eai_data/

# Delete the cloned repository to clean up
cd "$base_path"
rm -rf emergent-misalignment
