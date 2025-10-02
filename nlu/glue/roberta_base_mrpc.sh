WANDB_PROJECT="lorta-glue"
for rank in 1 2 4 8 16 32 64 128 256
do
    echo "Running with rank $rank"
    CUDA_VISIBLE_DEVICES=0 python run_glue.py \
         --do_train \
         --do_eval \
         --gradient_accumulation_steps 1 \
         --output_dir ./output/model \
         --overwrite_output_dir \
         --logging_steps 10 \
         --logging_dir ./output/log \
         --eval_strategy epoch \
         --save_strategy epoch \
         --warmup_ratio 0.06 \
         --max_grad_norm 1000.0 \
         --weight_decay 0.0 \
         --shared_uv 1 \
         --shared_dim 768 \
         --model_name_or_path roberta-base \
         --per_device_train_batch_size 64 \
         --max_seq_length 128 \
         --mode lora \
         --lora_r ${rank} \
         --init_type 1 \
         --d_init_type 94 \
         --seed 42 \
         --task_name mrpc \
         --num_train_epochs 30 \
         --classifier_lr 2e-2 \
         --use_preconditioner \
         --preconditioner_epsilon 1e-6 \
         --learning_rate 1e-2
done
