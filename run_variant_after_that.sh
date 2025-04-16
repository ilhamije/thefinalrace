#!/bin/bash

# nohup ./run_variant_after_that.sh > wait_then_run.log 2>&1 &


# Wait until no more prev training is running
while pgrep -f "RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-a_noshift" > /dev/null; do
    echo "$(date): RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-a_noshift still running... waiting 5 mins..."
    sleep 300
done

# then run variant A
nohup bash tools/dist_train.sh configs/segnext/RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-b_shift3x3.py 4 \
    --work-dir work_dirs/RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-b_shift3x3 > \
    RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-b_shift3x3.log 2>&1 &

echo "$(date): Starting finetune var-b hp-b3 training..."