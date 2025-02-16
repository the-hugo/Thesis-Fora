#!/bin/bash

while true; do
    # Get GPU usage info
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    # Set your own thresholds (e.g., below 5% utilization and memory usage < 1000MiB)
    if [[ "$GPU_UTIL" -lt 5 ]] && [[ "$GPU_MEM" -lt 1000 ]]; then
        echo "GPU is free! Running annotator.py..."
        python3 /home/ra37qax/annotator.py
        exit 0
    fi

    echo "GPU busy: $GPU_UTIL% utilization, $GPU_MEM MiB memory used. Waiting..."
    sleep 30  # Wait 30 seconds before checking again
done
