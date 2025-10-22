import logging

import torch
import torch.multiprocessing as mp
import shutil

from transformers import AutoModel
import gc

from .arguments import Arguments
from .fsdp import FSdpPipeline, run_worker
from .preprocess import PreprocessPipeline
from .train import TrainPipeline, PromptTrainPipeline
import os
import wandb
from torch.utils.data import Subset
from datetime import datetime



def timestamp():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def run_task_prompting(task: str, args: Arguments) -> None:
    """
    Runs the training pipeline in single-process (non-distributed) mode.
    """
    
    logging.info(f"Running task {task} with prompting evaluation style.")  
    dataset = PreprocessPipeline().preprocess_data_for_prompting(task)
    
    logging.info(f"Dataset after prompting preprocessing: {dataset["train"][0]}")


    pipeline = PromptTrainPipeline(
        hulu_args=args, current_task=task, tokenizer_name=args.tokenizer_name
    )

    pipeline.set_tokenized_datasets(
        train_dataset=dataset["train"],
        dev_dataset=dataset["validation"],
        test_dataset=dataset["test"],
    )
    
    dataloader = pipeline.train_loader

    output = next(iter(dataloader))
    logging.info(f"Sample batch from train dataloader: {output}")

    metrics, avg_loss = pipeline.evaluate_generation(pipeline.dev_loader)

    # trained_model = pipeline.training()

    # avg_loss, metrics = pipeline.evaluate(trained_model, pipeline.dev_loader)

    # avg_loss = avg_loss.detach().cpu().item()
    
    pipeline.append_dict_to_csv(
        file_path=os.path.join(args.results_dir, f"results_{pipeline.current_task}.csv"),
        row_dict={
            "task": pipeline.current_task,
            "model_name": args.model_name,
            "train_epochs": args.train_epochs,
            "train_batch": args.train_batch,
            "train_lr": args.train_lr,
            "train_maxlen": args.train_maxlen,
            "precision": args.precision,
            "use_lora": args.use_lora,
            "eval_style": args.eval_style,
            "avg_val_loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "mcc": metrics["mcc"],
            "f1": metrics["f1"],
            "training_time": pipeline.training_time,
            "learning_rate_scheduler_type": args.scheduler_type,
            "warmup_steps": args.train_warmup,
            "num_fewshot": args.num_fewshot,
        }
    )

def write_out_GPU_memory(stage: str) -> None:
    gpu_idx = 0  # usually 0 if you have one GPU
    total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9  # in GB
    reserved_mem = torch.cuda.memory_reserved(gpu_idx) / 1e9
    allocated_mem = torch.cuda.memory_allocated(gpu_idx) / 1e9
    free_mem = reserved_mem - allocated_mem

    logging.info(f"GPU Memory at stage: {stage}")

    logging.info(f"Total GPU memory: {total_mem:.2f} GB")
    logging.info(f"Reserved memory: {reserved_mem:.2f} GB")
    logging.info(f"Allocated memory: {allocated_mem:.2f} GB")
    logging.info(f"Free within reserved: {free_mem:.2f} GB")

def run_task_standard(task: str, args: Arguments) -> None:
    """
    Runs the training pipeline in single-process (non-distributed) mode.
    """

    dataset = PreprocessPipeline().preprocess_dataset(args, task)
    
    temp_model_path = "temp_model.pth"
    best_val_loss = float("inf")
    best_run_idx = -1
    trained_model = None

    act_timestamp = timestamp()

    for run_idx in range(args.repeatable_runs):
        logging.info(f"Running task {task}, repeatable run {run_idx+1}/{args.repeatable_runs}.")

        write_out_GPU_memory(f"before training of task {task}, run_idx {run_idx}")

        pipeline = TrainPipeline(
            hulu_args=args, current_task=task, tokenizer_name=args.tokenizer_name
        )

    # Split validation dataset into validation and test (50/50) randomly
    # val_dataset = dataset["validation"]
    # val_dataset = val_dataset.train_test_split(test_size=0.5, seed=42)
    # dataset["validation"] = val_dataset["train"]
    # dataset["test"] = val_dataset["test"]

    # logging.info(f"Train dataset size: {dataset['test']}")

        pipeline.set_tokenized_datasets(
            train_dataset=dataset["train"],
            dev_dataset=dataset["validation"],
            test_dataset=dataset["test"],
        )

        trained_model = pipeline.training()

        write_out_GPU_memory(f"after training of task {task}")

        avg_loss, metrics = pipeline.evaluate(trained_model, pipeline.dev_loader)

        avg_loss = avg_loss.detach().cpu().item()

        if avg_loss < best_val_loss:
            
            best_val_loss = avg_loss
            best_run_idx = run_idx
            torch.save(trained_model.state_dict(), temp_model_path)
    
        pipeline.append_dict_to_csv(
            file_path=os.path.join(args.results_dir, f"results_{pipeline.current_task}.csv"),
            row_dict={
                "task": pipeline.current_task,
                "model_name": args.model_name,
                "train_epochs": args.train_epochs,
                "train_batch": args.train_batch,
                "train_lr": args.train_lr,
                "train_maxlen": args.train_maxlen,
                "precision": args.precision,
                "use_lora": args.use_lora,
                "eval_style": args.eval_style,
                "avg_val_loss": avg_loss,
                "accuracy": metrics["accuracy"],
                "mcc": metrics["mcc"],
                "f1": metrics["f1"],
                "training_time": pipeline.training_time,
                "learning_rate_scheduler_type": args.scheduler_type,
                "warmup_steps": args.train_warmup,
                "num_fewshot": "na",
                "timestamp": act_timestamp,
            }
        )
        if run_idx < args.repeatable_runs - 1:
            del trained_model, pipeline
            torch.cuda.empty_cache()
            gc.collect()

        write_out_GPU_memory(f"after run_idx {run_idx} of task {task}")

    best_model_path = f"/project/c_racka1/racka_komondor/saved_models/{args.model_name.replace('/', '-')}_{task}_run_idx{best_run_idx}\
    _lora_finetune_eval_learningrate_{args.train_lr}_scheduler_{args.scheduler_type}_eval_style_ \
    {args.eval_style}_warmup_{args.train_warmup}_timestamp_{act_timestamp}.pth"
    
    trained_model.load_state_dict(torch.load(temp_model_path, map_location=trained_model.device))
    torch.save(trained_model.state_dict(), best_model_path)
    os.remove(temp_model_path)
    
    
    # pipeline.create_submission(trained_model)


def run_task_fsdp(task: str, args: Arguments) -> None:
    """
    Runs the training pipeline using FSDP (distributed training).
    This spawns one process per available CUDA device.
    """
    dataset = PreprocessPipeline().preprocess_dataset(args, task)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("FSDP selected but no CUDA devices found.")

    mp.spawn(
        run_worker,
        args=(world_size, args, task, args.tokenizer_name, dataset),
        nprocs=world_size,
        join=True,
    )


def benchmark(args: Arguments) -> None:
    """
    For each task specified in the arguments, run the appropriate training pipeline.
    """
    
    if args.eval_style == "prompt":
        task_runner = run_task_prompting
    else:
        task_runner = run_task_fsdp if args.use_fsdp else run_task_standard

    api_key = os.getenv('WANDB_API_KEY', 'None')

    logging.info(api_key)

    wandb.login(key=api_key)
    run = wandb.init(
        project="Racka-eval", 
        entity='elte-dh', 
        name=f"{args.model_name}_lora_finetune_eval_learningrate_{args.train_lr}_scheduler_{args.scheduler_type}_eval_style_{args.eval_style}_warmup_{args.train_warmup}",
        config={
            "model_name": args.model_name,
            "train_epochs": args.train_epochs,
            "train_batch": args.train_batch,
            "train_lr": args.train_lr,
            "train_maxlen": args.train_maxlen,
            "precision": args.precision,
            "use_lora": args.use_lora,
            "eval_style": args.eval_style,
            "learning_rate_scheduler_type": args.scheduler_type,
            "learning_rate": args.train_lr,
            "warmup_steps": args.train_warmup,
        })

    args.wandb_run_id = run.id

    for task in args.tasks:
        logging.info(
            "######### Started evaluating %s on task %s", args.model_name, task
        )
        task_runner(task, args)
    run.finish()
