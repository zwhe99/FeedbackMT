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
config=$WS/train_scripts/config/nllb-200-distilled-600M.json
model_save=$WS/models/sft-data.wmt20-train-20M+llm.random-600M
data_path=$WS/data/sft/wmt20-train-20M-nllb.json

# HOST_NUM will be 1
torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --train_file $data_path \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --max_steps 500000 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "inverse_sqrt" \
    --logging_steps 10 \
    --do_train \
    --evaluation_strategy "no" \
    --fp16 True \
    --use_fast_tokenizer False \
    --ddp_timeout 3600 \
    --seed 1 \
    --output_dir ${model_save} \
    --from_scratch True \
    --config_name $config \
    --streaming True \
    --tokenizer_name facebook/nllb-200-1.3B
