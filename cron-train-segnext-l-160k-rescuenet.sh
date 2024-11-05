#!/bin/bash

# Define variables
LOG_FILE="train-segnext-l-160k-rescuenet.log"
CRON_LOG_FILE="cron-train-l-160k.log"

PROCESS_COMMAND="CUDA_VISIBLE_DEVICES=1 nohup python tools/train.py configs/segnext/segnext_mscan-l_1xb16-adamw-160k_rescuenet-512x512.py --resume > $LOG_FILE 2>&1 &"

# Function to get the current timestamp
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

echo "The bash script STARTED | $(timestamp)"
echo "The bash script STARTED | $(timestamp)" >> "$CRON_LOG_FILE"

while true
do
    # Check if the process is running
    if pgrep -f "python tools/train.py configs/segnext/segnext_mscan-l_1xb16-adamw-160k_rescuenet-512x512.py" > /dev/null
    then
        echo "Process is still running. | $(timestamp)"
        echo "Process is still running. | $(timestamp)" >> "$CRON_LOG_FILE"
    else
        # Check if 160000/160000 is in the last 100 lines of the log
        if ! tail -n 100 "$LOG_FILE" | grep -q "160000/160000"
        then
            echo "Process not running and hasn't reached 160000/160000. Restarting process... | $(timestamp)"
            echo "Process not running and hasn't reached 160000/160000. Restarting process... | $(timestamp)" >> "$CRON_LOG_FILE"
            eval $PROCESS_COMMAND
        else
            echo "PROCESS FINISHED | $(timestamp)"
            echo "PROCESS FINISHED | $(timestamp)" >> "$CRON_LOG_FILE"
        fi
    fi

    # Wait for 300 seconds before repeating
    sleep 300
done
