echo "Environment Variables:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  RANK: $RANK"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

export VLLM_ATTENTION_BACKEND=XFORMERS

TRAIN_PATH=/train
VAL_PATH=/val
MODEL_PATH=/model
SAVE_PATH=/save

ray start --head --port=6000

TRAINING_NODES=$WORLD_SIZE  
TRAINING_GPUS_PER_NODE=$NPROC_PER_NODE

echo "Training Configuration:"
echo "  TRAINING_NODES: $TRAINING_NODES"
echo "  TRAINING_GPUS_PER_NODE: $TRAINING_GPUS_PER_NODE"

echo "Starting training on the Master node..."

SYSTEM_PROMPT="""A conversation between User and Assistant.
The User provides an image and asks a question.
The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer.
The Assistant must carefully checkout the correctness and validity of each reasoning step.
If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
Between <answer> and </answer>, the key verifiable answer is enclosed within \\boxed{}. If it is a multiple choice question, put the choice letter (e.g. A, B, C, D, E, F) in the \\boxed{}.
For example, <think> detailed reasoning process here, with potential reflections and corrections </think><answer> final answer here, with key answer enclosed within \\boxed{} </answer>"""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_PATH} \
    data.val_files=${VAL_PATH} \
    data.seed=11 \
    data.rollout_batch_size=512 \
    data.system_prompt="$SYSTEM_PROMPT" \
    worker.actor.global_batch_size=128 \
    worker.actor.kl_loss_coef=1.0e-3 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1v+cir \
    worker.reward.cur_type=s_2 \
    trainer.n_gpus_per_node=$TRAINING_GPUS_PER_NODE \
    trainer.nnodes=$TRAINING_NODES \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.save_freq=10
