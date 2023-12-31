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

train_path=$WS/src/feedback_training_raft.py
model_path=$WS/models/sft-data.wmt20-train+llm.llama-2-7b-hf
model_save=$WS/models/raftplus-data.cc-28k+llm.llama-2-7b-hf+rm.da
data_path=$WS/data/raft/cc-28k
deepspeed_cfg=$WS/train_scripts/config/deepspeed_config_zero2_raft_llama2.json
log_dir=$model_save/log
reward_dir=$model_save/reward

mkdir -p $log_dir $reward_dir

deepspeed --num_gpus 8 --master_port $MASTER_PORT \
    $train_path \
    --len_ratio "{\"en-de\": \"1.18,1.58\", \"de-en\": \"0.68,0.92\", \"en-zh\": \"1.59,2.10\", \"zh-en\": \"0.51,0.67\"}" \
    --lang_detect True \
    --repeat_detect True \
    --use_fast_tokenizer False \
    --reward_type comet-qe \
    --reward_model_or_path anon0616/wmt21-comet-qe-da \
    --model_name_or_path $model_path \
    --num_raft_iteration 10 \
    --learning_rate 2e-6 \
    --lr_scheduler_type "constant" \
    --bf16 \
    --deepspeed $deepspeed_cfg \
    --dataset_path $data_path \
    --output_reward_path $reward_dir/reward.txt \
    --output_dir $model_save \
    --num_train_epochs 1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 2 \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 7777 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 12 \
    --inference_batch_size_per_device 2 \
    --collection_strategy "local" \
    --raft_batch_size 1024 \
    --output_min_length 1 \
    --output_max_length 256 \
    --top_reward_percentage 0.125 \
    | tee ${log_dir}/raft_align.log \
    2> ${log_dir}/raft_align.err
