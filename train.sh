#!/bin/bash

# Parameters arrays
sparsity_values=(0.5 0.7 0.8 0.9 0.95 0.98)

export METHOD=$1 # L1, RST, pretrain
export BATCH_SIZE=$2
export PRETRAIN_EXP_ID=$3

# Check if sparsity_values is not empty
if [ "$METHOD" == "L1" ]; then
    for sparsity in "${sparsity_values[@]}"; do
        # L1
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
                                            --project_name deit_base_patch16_224_L1_${sparsity} 2>&1 | tee ExpID.log
        EXP_ID=$(grep "EXP_ID:" ExpID.log | head -n 1 | awk -F':' '{print $2}')
        # LTH
        python -m torch.distributed.launch  --nproc_per_node=4 \
                                            --master_port=12345 \
                                            --use_env main.py \
                                            --model deit_base_patch16_224 \
                                            --batch-size $BATCH_SIZE \
                                            --data-path data/imagenet \
                                            --wg weight \
                                            --inherit_pruned index \
                                            --base_pr_model Experiments/deit_base_patch16_224_L1_${sparsity}_${EXP_ID}/weights/checkpoint_just_finished_prune.pth \
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
elif [ "$METHOD" == "pretrain" ]; then
    python -m torch.distributed.launch  --nproc_per_node=4 \
                                        --master_port=12345 \
                                        --use_env main.py \
                                        --model deit_base_patch16_224 \
                                        --batch-size $BATCH_SIZE \
                                        --data-path data/imagenet \
                                        --save_init_model \
                                        --project_name deit_base_patch16_224_pretrain
else
    echo "Unknown METHOD: $METHOD"
    exit 1
fi
