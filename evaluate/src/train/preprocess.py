import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .arguments import Arguments
from .constants import (
    CONJUNCTIONS,
    HULU_DATASETS,
    IRRELEVANT_COLUMNS,
    LABELS,
    RELEVANT_COLUMNS,
    TOKENIZER_PARAMETERS,
)
from huggingface_hub import login
import os


class PreprocessPipeline:
    def __init__(self):
        self.tokenizer = None
        self.tokenizer_params = None
        self.preprocess_fn = None

    def preprocess_dataset(self, arguments: Arguments, task: str):
        dataset_name = HULU_DATASETS[task]

        dataset = load_dataset(dataset_name)
        self.tokenizer, self.tokenizer_params = self.load_tokenizer(arguments, task)


        self.preprocess_fn = self.get_preprocess_fn(
            task, self.tokenizer, self.tokenizer_params
        )
        
        if arguments.eval_style == "prompting":
            dataset = dataset.map(lambda batch: self._map_labels_for_prompting(batch, task), batched=True)

        remove_columns = IRRELEVANT_COLUMNS.get(task)

        dataset = dataset.map(self.preprocess_fn, remove_columns=remove_columns)

        return dataset
    
    def _map_labels_for_prompting(batch, task):
        conversions = CONVERSIONS.get(task, [])
        if not conversions:
            return batch  # nothing to change

        # build a mapping for quick lookup
        conversion_dict = {c["from"]: c["to"] for c in conversions}

        # convert labels
        batch["label"] = [
            conversion_dict.get(label, label) for label in batch["label"]
        ]

        return batch
        

    def get_preprocess_fn(self, task, tokenizer, tokenizer_params):
        def preprocess_example(
            premise, choice1, choice2, question, tokenizer, tokenizer_params
        ):
            choices = [choice1, choice2]
            input_strings = [
                f"{premise} {CONJUNCTIONS[question]} {choice}" for choice in choices
            ]
            tokenized_choices = [
                tokenizer(
                    input_str,
                    truncation=tokenizer_params["truncation"],
                    max_length=tokenizer_params["max_length"],
                    padding=tokenizer_params["padding"],
                )
                for input_str in input_strings
            ]
            input_ids = torch.tensor([x["input_ids"] for x in tokenized_choices])
            attention_mask = torch.tensor(
                [x["attention_mask"] for x in tokenized_choices]
            )

            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def preprocess_row(row):
            label = LABELS.get(row.get("label"), row.get("label"))

            if task == "copa":
                return preprocess_example(
                    row.get("premise"),
                    row.get("choice1"),
                    row.get("choice2"),
                    row.get("question"),
                    tokenizer,
                    tokenizer_params,
                )

            tokenized = tokenizer(
                *tuple(row[col_name] for col_name in RELEVANT_COLUMNS[task]),
                **tokenizer_params,
            )

            return {"label": label, **tokenized}

        return preprocess_row

    def load_tokenizer(self, arguments: Arguments, task: str):
        token = os.getenv("HF_TOKEN")

        login(token=token)
        
        tokenizer = AutoTokenizer.from_pretrained(
            arguments.tokenizer_name, clean_up_tokenization_spaces=True
        )
        print(f"Downloaded tokenizer {arguments.tokenizer_name} from Hugging Face")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer_params = TOKENIZER_PARAMETERS[task]

        if task in ["cola", "sst", "wnli", "rte", "copa"]:
            tokenizer_params["max_length"] = arguments.train_maxlen

        return tokenizer, tokenizer_params
