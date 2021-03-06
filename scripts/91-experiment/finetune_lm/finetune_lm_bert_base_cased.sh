CUDA_VISIBLE_DEVICES=1,6,7 python -m torch.distributed.launch --nproc_per_node 3 run_language_modeling.py   \
    --train_data_file "data/SemEval-2020-Task5/lm_finetuning_all_sentences.csv" \
    --output_dir "random_split_output/SemEval-2020-Task5/bert-base-cased/pretrained_language_model" \
    --model_type bert \
    --model_name_or_path "/users5/kliao/manual_cache/pytorch_transformers/bert-base-cased" \
    --mlm \
    --line_by_line \
    --do_train \
    --fp16  \
    --fp16_opt_level O2  \
    --block_size 128 \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --overwrite_output_dir   \
    --overwrite_cache \