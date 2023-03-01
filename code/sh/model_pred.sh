# batch size 12 for 16 GB GPU

mnt_dir="/content"

# You may change the following block for multiple gpu training
MASTER_HOST=localhost
MASTER_PORT=23333
RANK=0
PER_NODE_GPU=1
WORLD_SIZE=1
NODES=1
NCCL_DEBUG=INFO

#bash test_nltk.sh


# Change the arguments as required:
#   model_name_or_path, load_model_path: the path of the model to be finetuned
#   eval_file: the path of the evaluation data
#   output_dir: the directory to save finetuned model (not used at infer/test time)
#   out_file: the path of the output file
#   train_file_name: can be a directory contraining files named with "train*.jsonl"

python /content/CodeReviewerFineTuned/code/model_pred.py  \
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
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233