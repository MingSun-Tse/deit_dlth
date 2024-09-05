#!/bin/bash

# Parameters arrays
sparsity_values=(0.5 0.7 0.8 0.9 0.95 0.98)

# export OPT=sgd
# export SCHEDULER=step
# export LEARNING_RATE=""
export METHOD="RST" # L1, LTH, RST
export PRETRAIN_EXP_ID="deit_base_patch16_224_pretrain_SERVER-20240904-185102" # L1, LTH, RST
export PRUNED_EXP_ID="" # LTH
export BATCH_SIZE=256

if [[ -n "$METHOD" ]]; then
    if [[ -z "$PRETRAIN_EXP_ID" ]]; then
        echo "ERROR: METHOD is defined, but PRETRAIN_EXP_ID is not set."
        exit 1
    fi

    if [ "$METHOD" == "LTH" ]; then
        if [[ -z "$PRUNED_EXP_ID" ]]; then
            echo "ERROR: METHOD is set as LTH, but PRUNED_EXP_ID is not set."
            exit 1
        fi
    fi

    if [ ${#sparsity_values[@]} -eq 0 ]; then
        echo "ERROR: METHOD is defined, but sparsity_values is empty."
        exit 1
    fi

    # if [[ -z "$LEARNING_RATE" ]]; then
    #     echo "ERROR: METHOD is defined, but LEARNING_RATE is not set."
    #     exit 1
    # fi
fi

# Check if sparsity_values is not empty
if [ ${#sparsity_values[@]} -gt 0 ]; then
    if [ "$METHOD" == "L1" ]; then
        # L1
        for sparsity in "${sparsity_values[@]}"; do
            python -m torch.distributed.launch  --nproc_per_node=4 \
                                                --master_port=12345 \
                                                --use_env main.py \
                                                --model deit_base_patch16_224 \
                                                --batch-size $BATCH_SIZE \
                                                --data-path data/imagenet \
                                                --stage_pr "[0,$sparsity,0]" \
                                                --wg weight \
                                                --method L1 \
                                                --base_model_path Experiments/${PRETRAIN_EXP_ID}/weights/checkpoint.pth \
                                                --project_name deit_base_patch16_224_L1_${sparsity}
        done
    elif [ "$METHOD" == "LTH" ]; then
        # LTH
        for sparsity in "${sparsity_values[@]}"; do
            python -m torch.distributed.launch  --nproc_per_node=4 \
                                                --master_port=12345 \
                                                --use_env main.py \
                                                --model deit_base_patch16_224 \
                                                --batch-size $BATCH_SIZE \
                                                --data-path data/imagenet \
                                                --wg weight \
                                                --inherit_pruned index \
                                                --base_pr_model Experiments/${PRUNED_EXP_ID}/weights/checkpoint_just_finished_prune.pth \
                                                --method L1 \
                                                --base_model_path Experiments/${PRETRAIN_EXP_ID}/weights/checkpoint_init.pth \
                                                --project_name deit_base_patch16_224_LTH_${sparsity}
        done
    elif [ "$METHOD" == "RST" ]; then
        # RST
        for sparsity in "${sparsity_values[@]}"; do
            python -m torch.distributed.launch  --nproc_per_node=4 \
                                                --master_port=12345 \
                                                --use_env main.py \
                                                --model deit_base_patch16_224 \
                                                --batch-size $BATCH_SIZE \
                                                --data-path data/imagenet \
                                                --wg weight \
                                                --method RST \
                                                --base_model_path Experiments/${PRETRAIN_EXP_ID}/weights/checkpoint_init.pth \
                                                --stage_pr "[0,$sparsity,0]" \
                                                --project_name deit_base_patch16_224_RST_${sparsity}
        done
    else
        echo "Unknown METHOD: $METHOD"
        exit 1
    fi
else
    python -m torch.distributed.launch  --nproc_per_node=4 \
                                        --master_port=12345 \
                                        --use_env main.py \
                                        --model deit_base_patch16_224 \
                                        --batch-size $BATCH_SIZE \
                                        --data-path data/imagenet \
                                        --save_init_model \
                                        --project_name deit_base_patch16_224_pretrain
fi