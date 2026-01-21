#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 定义通用参数
TASK="long_term_forecast"
IS_TRAINING=1
ROOT_PATH="./dataset/ETT-small"
FEATURES="S"
E_LAYERS=1
D_LAYERS=1
ENC_IN=1
DEC_IN=1
C_OUT=1
DES="'Exp'"
ITR=1

# 定义函数执行模型
run_model() {
    local model=$1
    local pred_len=$2
    local seq_len=$3
    local dataset=$4
    local data_path="${dataset}.csv"
    local extra_args=""
    
    # 设置label_len等于pred_len
    local label_len=48
    
    # 动态生成model_id，包含数据集名称、seq_len和pred_len
    local model_id="${dataset}_${seq_len}_${pred_len}"
    
    # 处理特殊模型的参数
    case $model in
        "iTransformer")
            extra_args="--d_model 16 --d_ff 16"
            ;;
        "MICN")
            extra_args="--factor 3"
            ;;
        "TimesNet")
            extra_args="--d_model 8 --d_ff 16 --top_k 1"
            ;;
    esac

    echo "运行模型: $model, 数据集: $dataset, 预测长度: $pred_len, 序列长度: $seq_len, 标签长度: $label_len"
    
    # 执行命令
    python -u run.py \
      --task_name $TASK \
      --is_training $IS_TRAINING \
      --root_path $ROOT_PATH \
      --data_path $data_path \
      --model_id $model_id \
      --model $model \
      --data $dataset \
      --features $FEATURES \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers $E_LAYERS \
      --d_layers $D_LAYERS \
      --enc_in $ENC_IN \
      --dec_in $DEC_IN \
      --c_out $C_OUT \
      --embed cftode \
      --des $DES \
      --itr $ITR \
      $extra_args
}

# 定义要运行的模型列表
MODELS=(
    'TimesNet'
    'Autoformer'
    'Transformer'
    'Nonstationary_Transformer'
    'FEDformer'
    'Informer'
    'Reformer'
    'ETSformer'
    'Pyraformer'
    'iTransformer'
    'TSMixer'
    'TemporalFusionTransformer'
    "SCINet"
    'PAttn'
    'TimeXer'
)

# 定义要运行的seq_len和pred_len组合
COMBINATIONS=(
    "96 240"
    "96 336"
    "96 720"
    "96 960"
)

# 定义要运行的数据集
DATASETS=(
    "ETTh1"
    "ETTh2"
    "ETTm1"
    "ETTm2"
)

# 主循环
for dataset in "${DATASETS[@]}"; do
    for combination in "${COMBINATIONS[@]}"; do
        # 分割组合字符串获取pred_len和seq_len
        read seq_len pred_len <<< "$combination"
        
        for model in "${MODELS[@]}"; do
            run_model "$model" "$pred_len" "$seq_len" "$dataset"
        done
    done
done