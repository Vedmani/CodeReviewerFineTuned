bash test_nltk.sh

python -m /content/CodeReviewerFineTuned/code/dummy_run_finetune_cls.py  \
  --train_epochs 30 \
  --model_name_or_path /content/model \
  --output_dir /content/finetuned_model \
  --train_filename /content/data/Diff_Quality_Estimation \
  --dev_filename /content/data/Diff_Quality_Estimation/cls-valid.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --train_batch_size 8 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 3 \
  --mask_rate 0.15 \
  --save_steps 3600 \
  --log_steps 100 \
  --train_steps 120000 \
  --seed 2233