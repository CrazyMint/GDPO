import os


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
        os.environ["WANDB_RESUME"] = "allow"
        ## THIS IS JUST HARD CODED TO MAKE IT WANDB RESUME RUN WORKS
        # if training_args.run_name == "Qwen2.5-1.5B-Open-R1-GRPO":
        #     os.environ["WANDB_RUN_ID"] = "87lyx07v"
        # if training_args.run_name == "Qwen2.5-1.5B-Open-R1-GDPO":
        #     os.environ["WANDB_RUN_ID"] = "88lyz87z"
        # if training_args.run_name == "DeepSeek-R1-Distill-Qwen-1.5B-GRPO":
        #     os.environ["WANDB_RUN_ID"] = "89lyx07v"
        # if training_args.run_name == "DeepSeek-R1-Distill-Qwen-1.5B-GDPO":
        #     os.environ["WANDB_RUN_ID"] = "90lyz87z"
        # if training_args.run_name == "Qwen2.5-1.5B-gsm8k-GRPO":
        #     os.environ["WANDB_RUN_ID"] = "90lyz64z"
        # if training_args.run_name == "Qwen2.5-1.5B-gsm8k-GDPO":
        #     os.environ["WANDB_RUN_ID"] = "97lyz64z"
        # if training_args.run_name == "Qwen2.5-3B-gsm8k-GRPO":
        #     os.environ["WANDB_RUN_ID"] = "93lyz64z"
        # if training_args.run_name == "Qwen2.5-3B-gsm8k-GDPO":
        #     os.environ["WANDB_RUN_ID"] = "92lyz64z"

    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
