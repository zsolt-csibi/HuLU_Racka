import evaluate
import numpy as np
import torch
from torch import amp, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerFast,
)

from transformers.modeling_outputs import (
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)

from safetensors.torch import load_file
from peft import PeftModel

from .arguments import Arguments
from .constants import CB_LABELS, CONJUNCTIONS, SST_LABELS
from .helper import write_submission
from .lora_helper import set_lora

accuracy_metric = evaluate.load("accuracy")
mcc_metric = evaluate.load("matthews_correlation")
f1_metric = evaluate.load("f1")


def load_custom_embeddings_qwen(model,tokenizer, embed_path, lm_head_path):
    """Load custom embedding layers"""
    # Load modified embed_token
    embed_tokens_state_dict = load_file(embed_path)
    print(embed_tokens_state_dict.keys())
    # embed_tokens_state_dict = {"weight": embed_tokens_state_dict["model.embed_tokens.weight"]}
    embed_tokens = torch.nn.Embedding(151936, 2560, tokenizer.pad_token_id, dtype=torch.bfloat16)
    print(embed_tokens_state_dict["weight"].shape)
    print(embed_tokens_state_dict["weight"].dtype)

    embed_tokens.load_state_dict(embed_tokens_state_dict)
    model.embed_tokens = embed_tokens.bfloat16().to('cuda')

    torch.cuda.empty_cache()
    
    return model

def load_base_model(base_model_id, task, eval_style, model_kwargs={}):
    if task == "copa":
        model = AutoForMultipleChoice(base_model_id)
    elif eval_style == "standard":
        model = AutoModelForSequenceClassification.from_pretrained(base_model_id,
            attn_implementation="sdpa",
            device_map="auto", **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            attn_implementation="sdpa",
             device_map="auto"
    )

    print(f"Model loaded: {base_model_id}")

    return model

def load_local_model(model_id, model_path, task, eval_style, tokenizer, apply_quantization=False, model_kwargs={}):
    """Load and configure the base model"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_base_model(model_id, task, eval_style, model_kwargs)

    embed_path = model_path

    if task == "copa":
        model.to(device)
        model.model.gradient_checkpointing_enable({"use_reentrant":False})
        model.model = load_custom_embeddings_qwen(model.model, tokenizer, f"{embed_path}/embed_tokens.safetensors", lm_head_path=embed_path)
        print(f"Custom embeddings loaded: {embed_path} for task {task}")

    # Apply PEFT adapters
        model.model = PeftModel.from_pretrained(model.model, model_path)

        print(f"PEFT adapters loaded from: {model_path} for task {task}")

        model.model = model.model.merge_and_unload()
    else:
        if eval_style == "standard":
            model.to(device)
            model.gradient_checkpointing_enable({"use_reentrant":False})
            model = load_custom_embeddings_qwen(model, tokenizer, f"{embed_path}/embed_tokens.safetensors", lm_head_path=embed_path)
            print(f"Custom embeddings loaded: {embed_path} for task {task}")

    # Apply PEFT adapters
            model.base_model = PeftModel.from_pretrained(model.base_model, model_path)

            print(f"PEFT adapters loaded from: {model_path} for task {task}")

            model.base_model = model.base_model.merge_and_unload()
        else:
            model.to(device)
            model.gradient_checkpointing_enable({"use_reentrant":False})
            model = load_custom_embeddings_qwen(model, tokenizer, f"{embed_path}/embed_tokens.safetensors", lm_head_path=embed_path)
            print(f"Custom embeddings loaded: {embed_path} for task {task}")

    # Apply PEFT adapters
            model = PeftModel.from_pretrained(model, model_path)
            print(f"PEFT adapters loaded from: {model_path} for task {task}")
            model = model.merge_and_unload()

    return model

def load_local_tokenizer(tokenizer_path, max_length=4096):
    """Load and configure tokenizer"""
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, trust_remote_code=True)

    # Add all special tokens consistently with tokenize_test.py
    special_tokens = {
        "bos_token": "<|endoftext|>",
        "eos_token" : "<|endoftext|>",
        "pad_token" : "<|endoftext|>",
    }
    tokenizer.add_special_tokens(special_tokens)

    print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")
    print(f"Tokenizer eos token ID: {tokenizer.eos_token_id}")
    tokenizer.model_max_length = max_length
    return tokenizer


def compute_metrics(logits, labels):
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    accuracy = accuracy_metric.compute(predictions=logits, references=labels)
    f1 = f1_metric.compute(predictions=logits, references=labels, average="weighted")
    mcc = mcc_metric.compute(predictions=logits, references=labels)

    return {
        "accuracy": accuracy["accuracy"],
        "mcc": mcc["matthews_correlation"],
        "f1": f1["f1"],
    }


class AutoForMultipleChoice(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, base_model_id,
            attn_implementation="sdpa",
            device_map="auto"
        )

        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, num_choices, seq_length = input_ids.shape
        input_ids = input_ids.view(-1, seq_length)

        attention_mask = (
            attention_mask.view(-1, seq_length) if attention_mask is not None else None
        )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        pooled_output = outputs.hidden_states[-1][:, -1, :]

        logits = self.classifier(pooled_output).view(batch_size, num_choices)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return MultipleChoiceModelOutput(loss=loss, logits=logits)
        return MultipleChoiceModelOutput(logits=logits)


class TrainPipeline:
    def __init__(self, hulu_args: Arguments, current_task: str, tokenizer_name: str):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.current_task = current_task
        self.hulu_args = hulu_args

        if self.hulu_args.local_model:
            self.tokenizer = load_local_tokenizer(self.hulu_args.tokenizer_name, self.hulu_args.train_maxlen)
            print(f"Local tokenizer loaded from {self.hulu_args.tokenizer_name}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, clean_up_tokenization_spaces=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

            # self.tokenizer.add_special_tokens(special_tokens)
            print(f"Tokenizer loaded from {tokenizer_name}")

        print("Tokenizer special tokens:", self.tokenizer.special_tokens_map)

        self.train_loader, self.dev_loader, self.test_loader = None, None, None

    def collate_fn(self, batch):
        for item in batch:
            item["input_ids"] = torch.tensor(item["input_ids"], dtype=torch.long)
            item["attention_mask"] = torch.tensor(
                item["attention_mask"], dtype=torch.long
            )
            if "label" in item:
                item["label"] = torch.tensor(item["label"], dtype=torch.long)

        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        if not "label" in batch[0]:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        labels = (
            torch.stack([item["label"] for item in batch])
            if "label" in batch[0]
            else None
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }

    def set_tokenized_datasets(self, train_dataset, dev_dataset, test_dataset):
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.hulu_args.train_batch,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.hulu_args.train_batch,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        test_dataset = test_dataset.remove_columns("label")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.hulu_args.train_batch,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def load_model(self):
        model_kwargs = (
            {"num_labels": 3}
            if self.current_task in ["sst", "cb"]
            else {"num_labels": 2}
        )
        if self.hulu_args.local_model:
            model = load_local_model(self.hulu_args.model_name, self.hulu_args.model_path, self.current_task, 
                self.hulu_args.eval_style, self.tokenizer, apply_quantization=False)
        else:
            if self.current_task == "copa":
                model = AutoForMultipleChoice(self.hulu_args.model_name)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.hulu_args.model_name, **model_kwargs
                )
        model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.hulu_args.use_lora:
            model = set_lora(self.hulu_args, sequente_classification=False, model=model)

        model.to(self.device)
        return model

    def training(self):
        model = self.load_model()

        use_fp16 = self.hulu_args.precision == "fp16"
        scaler = amp.GradScaler() if use_fp16 else None
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        optimizer = AdamW(model.parameters(), lr=self.hulu_args.train_lr)
        total_steps = len(self.train_loader) * self.hulu_args.train_batch
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hulu_args.train_warmup,
            num_training_steps=total_steps,
        )

        num_eval_steps = len(self.train_loader) // 3
        step = 0

        for epoch in range(self.hulu_args.train_epochs):
            model.train()
            total_loss, correct_preds = 0, 0

            for batch in tqdm(
                self.train_loader,
                desc=f"Training Epoch {epoch + 1}/{self.hulu_args.train_epochs}",
            ):
                step += 1
                input_ids, attention_mask, labels = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["label"].to(self.device),
                )

                optimizer.zero_grad()

                with amp.autocast(device_type=device_type, enabled=use_fp16):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                if use_fp16:
                    scaler.scale(output.loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output.loss.backward()
                    optimizer.step()

                scheduler.step()
                total_loss += output.loss.item()
                correct_preds += (output.logits.argmax(dim=1) == labels).sum().item()

                if step % num_eval_steps == 0:
                    eval_loss, metrics = self.evaluate(model)
                    print(
                        f"Step {step}: Eval Loss = {eval_loss:.4f}, Eval Acc = {metrics['accuracy']}, Eval MCC = {metrics['mcc']}, Eval F1 = {metrics['f1']}"
                    )

            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct_preds / len(self.train_loader.dataset)
            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.4f}"
            )

        return model

    def evaluate(self, model):
        model.eval()
        total_loss, correct_preds = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for _, batch in enumerate(self.dev_loader):
                input_ids, attention_mask, labels = (
                    batch["input_ids"].squeeze(1).to(self.device),
                    batch["attention_mask"].squeeze(1).to(self.device),
                    batch["label"].to(self.device),
                )

                output = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                total_loss += output.loss
                preds = output.logits.argmax(dim=1)
                correct_preds += (preds == labels).sum().item()

                all_preds.append(preds)
                all_labels.append(labels)

        avg_loss = total_loss / len(self.dev_loader)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        return avg_loss, metrics

    def create_submission(self, model):
        model.eval()

        if self.current_task == "sst":
            reverse_labels = {v: k for k, v in SST_LABELS.items()}
        elif self.current_task == "cb":
            reverse_labels = {v: k for k, v in CB_LABELS.items()}
        else:
            reverse_labels = None

        predictions = []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, attention_mask = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                output: MultipleChoiceModelOutput | SequenceClassifierOutput = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                predictions.extend(output.logits.argmax(dim=-1).tolist())

        predictions_data = [
            {
                "id": str(i),
                "label": reverse_labels[pred] if reverse_labels else str(pred),
            }
            for i, pred in enumerate(predictions)
        ]

        write_submission(
            task=self.current_task,
            predictions_data=predictions_data,
            output_dir=self.hulu_args.output_dir,
        )
