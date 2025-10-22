from xml.parsers.expat import model
import evaluate
import numpy as np
import shutil
import torch
from torch import amp, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerFast,
)

from .constants import (
    TOKENIZER_PARAMETERS,
    CB_PROMPT,
    WNLI_PROMPT,
    COPA_PROMPT_EFFECT,
    COPA_PROMPT_CAUSE,
    RTE_PROMPT,
    SST_PROMPT,
    COLA_PROMPT,
    INVERSE_CONVERSIONS
)

import copy

import time
import pandas as pd
import os
import wandb
import logging



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
import re

accuracy_metric = evaluate.load("accuracy")
mcc_metric = evaluate.load("matthews_correlation")
f1_metric = evaluate.load("f1")



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
        self.model = AutoModel.from_pretrained(model_name,
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
        self.training_time = ""

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
            batch_size=self.hulu_args.train_batch if self.current_task != "copa" else 1,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.hulu_args.train_batch if self.current_task != "copa" else 1,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        # test_dataset = test_dataset.remove_columns("label")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.hulu_args.train_batch if self.current_task != "copa" else 1,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def load_model(self):
        model_kwargs = (
            {"num_labels": 3}
            if self.current_task in ["sst", "cb"]
            else {"num_labels": 2}
        )
        if self.current_task == "copa":
            model = AutoForMultipleChoice(self.hulu_args.model_name)
            model.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.hulu_args.model_name, **model_kwargs
            )
            model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.hulu_args.use_lora:
            model = set_lora(self.hulu_args, sequente_classification=False, model=model)

        model.to(self.device)
        return model
    
    def append_dict_to_csv(self, file_path: str, row_dict: dict):
        """
        Load a CSV file (or create it if it doesn't exist), 
        append a row from a dictionary, and save it back.

        Parameters:
            file_path (str): Path to the CSV file.
            row_dict (dict): Dictionary with column-value pairs to append.
        """
        try:
            if os.path.exists(file_path):
                # Load existing CSV
                df = pd.read_csv(file_path, sep=';')

                # Append dictionary as a new row
                df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

            else:
                # Create a new DataFrame with the dictionary
                df = pd.DataFrame([row_dict])

            # Save updated (or new) CSV
            df.to_csv(file_path, sep=';',index=False)

            print("Row appended successfully!")

        except Exception as e:
            print(f"Error: {e}")

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

        start = time.time()

        run = wandb.init(project="Racka-eval", id=self.hulu_args.wandb_run_id, resume="allow")

        best_eval_loss = float("inf")
        # patience_counter = 0
        # patience = getattr(self.hulu_args, "early_stopping_patience", 3)  # default 3 evals
        temp_model = "./current_best_model.pth"

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
                    eval_loss, metrics = self.evaluate(model, self.dev_loader)
                    print(
                        f"Step {step}: Eval Loss = {eval_loss:.4f}, Eval Acc = {metrics['accuracy']}, Eval MCC = {metrics['mcc']}, Eval F1 = {metrics['f1']}"
                    )
                    run.log({
                        f"{self.current_task}/eval_loss": eval_loss,
                        f"{self.current_task}/eval_accuracy": metrics["accuracy"],
                        f"{self.current_task}/eval_mcc": metrics["mcc"],
                        f"{self.current_task}/eval_f1": metrics["f1"],
                    })
                    # Check early stopping
                    if eval_loss < best_eval_loss:
                        # Save the best model
                        print(f"New best model found at step {step} with eval loss {eval_loss:.4f}")
                        best_eval_loss = eval_loss
                        torch.save(model.state_dict(), "current_best_model.pth")
                        
                        # patience_counter = 0
                    #     logging.info(f"New best model found at step {step} with eval loss {eval_loss:.4f}")
                    #     best_eval_loss = eval_loss
                    #     patience_counter = 0
                    #     torch.save(model.state_dict(), "current_best_model.pth")
                
            # if patience_counter >= patience:
            #     break
            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct_preds / len(self.train_loader.dataset)
            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.4f}"
            )
            run.log({
                f"{self.current_task}/train_loss": avg_loss,
                f"{self.current_task}/train_accuracy": accuracy,
                f"{self.current_task}/learning_rate": scheduler.get_last_lr()[0],
            })

        elapsed_time = time.time() - start

        print(f"Best model reloaded from eval with loss {best_eval_loss:.4f}")
        model.load_state_dict(torch.load("current_best_model.pth", map_location=model.device))
        # model = AutoModel.from_pretrained(temp_model_path, device_map="auto")

        
    
        # torch.save(model.state_dict(), f"/project/c_racka1/racka_komondor/src/eval/HuLU_Racka/saved_models/{self.hulu_args.model_name.replace('/', '-')}_{self.current_task}_lora_finetune_eval_learningrate_{self.hulu_args.train_lr}_scheduler_{self.hulu_args.scheduler_type}_eval_style_{self.hulu_args.eval_style}_warmup_{self.hulu_args.train_warmup}.pth")
        os.remove("current_best_model.pth")

        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.training_time = f"{hours}h:{minutes}m:{seconds}s"

        return model

    def evaluate(self, model, dataloader):
        model.eval()
        total_loss, correct_preds = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
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

        avg_loss = total_loss / len(dataloader)

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


class PromptTrainPipeline(TrainPipeline):
    def __init__(self, hulu_args: Arguments, current_task: str, tokenizer_name: str):
        super().__init__(hulu_args, current_task, tokenizer_name)
        self.tokenizer_params = TOKENIZER_PARAMETERS[current_task]
        
            
    
    def _load_model_for_inferecence(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.hulu_args.model_name,
            trust_remote_code=True,
            load_in_8bit=False,
            torch_dtype=torch.float16 if self.hulu_args.precision == "fp16" else torch.float32,
        ).to(self.device)
        
        return model


    def _build_prompt(self, example):
        
        if self.current_task == "cola":
            prompt = COLA_PROMPT.format(text=example["sentence"])
        elif self.current_task == "sst":
            prompt = SST_PROMPT.format(text=example["sentence"])
        elif self.current_task == "rte":
            prompt = RTE_PROMPT.format(
                premise=example["premise"], hypothesis=example["hypothesis"]
            )
        elif self.current_task == "cb":
            prompt = CB_PROMPT.format(
                premise=example["premise"],
                hypothesis=example["hypothesis"],
            )
        elif self.current_task == "wnli":
            prompt = WNLI_PROMPT.format(
                sentence1=example["sentence1"], sentence2=example["sentence2"]
            )
        elif self.current_task == "copa":
            if example["question"] == "effect":
                prompt = COPA_PROMPT_EFFECT.format(
                    premise=example["premise"],
                    choice1=example["choice1"],
                    choice2=example["choice2"],
                    examples=""
                )
            else:
                prompt = COPA_PROMPT_CAUSE.format(
                    premise=example["premise"],
                    choice1=example["choice1"],
                    choice2=example["choice2"],
                    examples=""
                )
        else:
            raise ValueError(f"Unknown task: {self.current_task}")

        return prompt
        

    def collate_fn(self, batch):
        
        texts = [self._build_prompt(item) for item in batch]
        labels = [str(item["label"]) for item in batch]

        inputs = self.tokenizer(
            texts,
            truncation=self.tokenizer_params["truncation"],
            max_length=self.tokenizer_params["max_length"],
            padding=self.tokenizer_params["padding"],
            return_tensors="pt",
            add_special_tokens=self.tokenizer_params.get("add_special_tokens", True),
        )
        labels = self.tokenizer(
            labels,
            truncation=True,
            max_length=10,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]

        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels}
    
    
    def set_tokenized_datasets(self, train_dataset, dev_dataset, test_dataset):
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.hulu_args.train_batch if self.current_task != "copa" else 1,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.hulu_args.train_batch if self.current_task != "copa" else 1,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        # test_dataset = test_dataset.remove_columns("label")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.hulu_args.train_batch if self.current_task != "copa" else 1,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def evaluate_generation(self, dataloader):
        model = self._load_model_for_inferecence()
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)

                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Generate predictions
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract generated text (remove input prompt)
                generated_ids = outputs[:, input_ids.shape[1]:]
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # logging.info(f"Generated text: {generated_texts[0]}")
                
                # Extract true labels
                label_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # Convert text predictions to numerical labels
                for pred_text, true_text in zip(generated_texts, label_texts):
                    pred_label = self._text_to_label(pred_text.strip())
                    true_label = self._text_to_label(true_text.strip())
                    
                    all_preds.append(pred_label)
                    all_labels.append(true_label)
        
        # Convert to tensors for metric computation
        
        logging.info(f"All predictions: {all_preds}")
        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        
        metrics = compute_metrics(all_preds, all_labels)
        return metrics, 'na'

    def _text_to_label(self, text):
        text = text.lower().strip()
        conversions = INVERSE_CONVERSIONS.get(self.current_task, [])
        if not conversions:
            try:
                return int(text)
            except ValueError:
                return text  # return as is if conversion fails

        # build a mapping for quick lookup
        conversion_dict = {c["from"]: c["to"] for c in conversions}

        cleaned_text = text

        for key, value in conversion_dict.items():
            # Clean key and text for comparison
            if key in cleaned_text:
                cleaned_text = key
                break
        

        return conversion_dict.get(cleaned_text, -1)  # return original text if no match

