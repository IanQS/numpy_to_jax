#!/bin/bash

# List of directories to skip
SKIP_DIRS=("ipynb_checkpoints")

# Function to check if a directory should be skipped
should_skip() {
    local dir="$1"
    for skip_dir in "${SKIP_DIRS[@]}"; do
        if [[ "$dir" == *"$skip_dir"* ]]; then
            return 0
        fi
    done
    return 1
}

# Function to clear outputs of Jupyter notebooks
clear_notebook_outputs() {
    local dir="$1"
    for file in "$dir"*.ipynb; do
        if [[ -f "$file" ]]; then
            echo "Clearing outputs in $file"
            jupyter nbconvert --clear-output --inplace "$file" &
        fi
    done
    wait
}

# Recursively process directories
process_directory() {
    local dir="$1"
    if should_skip "$dir"; then
        echo "Skipping directory $dir"
        return
    fi

    clear_notebook_outputs "$dir"

    for subdir in "$dir"/*/; do
        if [[ -d "$subdir" ]]; then
            process_directory "$subdir"
        fi
    done
}

# Start processing from the current directory
process_directory "$(pwd)"

# Allow push to proceed
exit 0

