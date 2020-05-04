CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node 7 run_transformers.py   \
    --task_name semeval-2020-task5-subtask1 \
    --data_dir "data/SemEval-2020-Task5/" \
    --model_type bert \
    --model_name_or_path "/users5/kliao/manual_cache/pytorch_transformers/bert-large-cased" \
    --output_dir "output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/model" \
    --do_train \
    --train_with_pseudo_labels \
    --max_seq_length 128   \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=12   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --overwrite_output_dir   \
    --overwrite_cache \