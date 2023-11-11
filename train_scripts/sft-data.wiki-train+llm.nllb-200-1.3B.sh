set -e
set -u

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

BASEDIR=$(realpath `dirname $0`)
WS=$BASEDIR/..

train_path=$WS/src/sft_nllb.py
model_path=facebook/nllb-200-1.3B
model_save=$WS/models/sft-data.wiki-train+llm.nllb-200-1.3B
data_path=$WS/data/sft/wiki-train-nllb.json
deepspeed_cfg=$WS/train_scripts/config/deepspeed_config_zero3.json

# HOST_NUM will be 1
torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed $deepspeed_cfg \
    --model_name_or_path ${model_path} \
    --train_file $data_path \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --do_train \
    --evaluation_strategy "no" \
    --fp16 True \
    --fp16_full_eval True \
    --use_fast_tokenizer False \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing True \
    --output_dir ${model_save}