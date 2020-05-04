export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export DATA_DIR="data/SemEval-2020-Task5/random_split/"
export NPROC_PER_NODE=7
export MAX_SEQ_LENGTH=128
export TRAIN_BATCH_SIZE=32
export LEARNING_RATE=2e-5
export NUM_TRAIN_EPOCHS=3.0
export MODEL_TYPE=xlnet
export MODEL_NAME_OR_PATH="/users5/kliao/manual_cache/pytorch_transformers/xlnet-base-cased"
export OUTPUT_DIR="random_split_output/SemEval-2020-Task5/xlnet-base-cased"
export K=10


#!/bin/bash
for i in $(seq 1 $K)
do
  echo 'Cross validation fold number:'
  echo $i
  python -m torch.distributed.launch --nproc_per_node $NPROC_PER_NODE run_transformers.py   \
    --task_name semeval-2020-task5-subtask1 \
    --data_dir  $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path  $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --do_train   \
    --do_eval   \
    --max_seq_length $MAX_SEQ_LENGTH   \
    --per_gpu_eval_batch_size 32   \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE   \
    --learning_rate $LEARNING_RATE   \
    --num_train_epochs $NUM_TRAIN_EPOCHS  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --cross_validation \
    --num_folds $K \
    --fold_number $i 
done

python -m tools.join_cross_validation_results \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --num_folds $K