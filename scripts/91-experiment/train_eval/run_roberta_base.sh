CUDA_VISIBLE_DEVICES=1,6,7 python -m torch.distributed.launch --nproc_per_node 3 run_transformers.py   \
    --task_name semeval-2020-task5-subtask1 \
    --data_dir "data/SemEval-2020-Task5/random_split/" \
    --model_type roberta \
    --model_name_or_path "/users5/kliao/manual_cache/pytorch_transformers/roberta-base" \
    --output_dir "random_split_output/SemEval-2020-Task5/roberta-base/" \
    --do_train   \
    --do_eval   \
    --fp16  \
    --fp16_opt_level O2  \
    --max_seq_length 128   \
    --per_gpu_eval_batch_size=48   \
    --per_gpu_train_batch_size=48   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --overwrite_output_dir   \
    --overwrite_cache \