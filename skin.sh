#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr  
MASTER_PORT=$((RANDOM % 101 + 20000))
export MASTER_PORT  
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "Rendezvous Endpoint: $MASTER_ADDR:$MASTER_PORT"

srun -p medai_llm --quotatype=spot \
--cpus-per-task=4 \
--gres=gpu:1 python main_rad.py \
--config ./configs/skin.yaml \
--dataset skin \
--output_dir /your_path/RAD_clean/outputs/skin50-rad-topk10-vision_res50-text_cbert \
--loss_ratio 0.1 \
--bert_model_name /your_path/huggingface/ClinicalBERT \
--guideline_path ./guideline/qwen_maxtoken2k_skincap50_4sources.jsonl \
--max_length 512 \
--embed_dim 768 \
--contrast_ratio_text 0.1 \
--contrast_ratio_vision 0.001 \
