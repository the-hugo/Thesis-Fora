#!/bin/bash

# Set the directory where your checkpoint files are stored.
CHECKPOINT_DIR="/home/ra37qax"  # <-- change this to your actual directory

while true; do
    # Change to the checkpoint directory.
    cd "$CHECKPOINT_DIR" || { echo "Directory not found: $CHECKPOINT_DIR"; exit 1; }

    # Get a list of files containing "checkpoint" in their names.
    checkpoint_files=(checkpoint*)
    if [ "${checkpoint_files[0]}" = "checkpoint*" ]; then
        echo "No checkpoint files found in $CHECKPOINT_DIR."
        sleep 30
        continue
    fi

    maxnum=-1
    maxfile=""

    # Loop over all matching files.
    for file in "${checkpoint_files[@]}"; do
        # Use a regex to extract the numeric part after "checkpoint" with optional '-' or '_' in between.
        if [[ "$file" =~ checkpoint[-_]?([0-9]+) ]]; then
            num=${BASH_REMATCH[1]}
            if (( num > maxnum )); then
                maxnum=$num
                maxfile=$file
            fi
        else
            echo "Skipping file '$file' as it does not match expected pattern."
        fi
    done

    # If no valid checkpoint files were found, wait and try again.
    if [ -z "$maxfile" ]; then
        echo "No valid checkpoint files with a numeric suffix found."
        sleep 30
        continue
    fi

    # Delete any checkpoint files that are not the one with the highest number.
    for file in "${checkpoint_files[@]}"; do
        if [ "$file" != "$maxfile" ]; then
            rm -f "$file"
            echo "Removed $file"
        fi
    done

    echo "Keeping $maxfile"
    # Wait for 30 seconds before checking again.
    sleep 30
done
