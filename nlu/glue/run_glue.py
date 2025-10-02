#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import json
import logging
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
import wandb
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from peft import LorTaConfig as LoraConfig
from peft import TaskType, get_peft_model


save_full_weights = False
push_weights = True
_dirs = ["weights", "output", "runs"]
for _dir in _dirs:
    if not os.path.exists(_dir):
        os.makedirs(_dir, exist_ok=True)


class GradientLogger:
    def __init__(self, model):
        self.model = model
        self.last_step = -1

    def log(
        self,
        state: TrainerState,
    ):
        if self.last_step >= state.global_step or state.global_step % 10 != 0:
            return
        total_norm = 0.0
        total_norm_lora = 0.0
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                if "lora" in n:
                    total_norm_lora += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        total_norm_lora = total_norm_lora ** (1.0 / 2)
        if True:  # torch.distributed.get_rank() == 0:
            wandb.log({"gradient_norm": total_norm})
            wandb.log({"gradient_norm_lora": total_norm_lora})
        self.last_step = state.global_step


class WeightLogCallback(TrainerCallback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_step = -1

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        return


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    job_id: Optional[str] = field(default=None)
    wandb_run_group: Optional[str] = field(
        default=None,
    )
    mode: Optional[str] = field(
        default=None,
    )
    submode: Optional[str] = field(
        default=None,
    )
    wandb_run_name: Optional[str] = field(
        default=None,
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    d_init_type: Optional[int] = field(
        default=0,
    )
    identity_init: Optional[int] = field(
        default=0,
    )
    finetune_classifier: Optional[int] = field(
        default=0,
    )
    shared_uv: Optional[int] = field(
        default=0,
    )
    wandb_offline: Optional[int] = field(
        default=0,
    )
    norm_penalty: Optional[int] = field(
        default=0,
    )
    norm_alpha: Optional[float] = field(
        default=0.001,
    )
    use_float64: Optional[int] = field(
        default=0,
    )
    optimized_order: Optional[int] = field(
        default=0,
    )
    order: Optional[int] = field(
        default=0,
    )
    shared_d: Optional[int] = field(
        default=0,
    )
    classifier_lr: Optional[float] = field(
        default=4e-4,
    )
    classifier_wd: Optional[float] = field(
        default=0.1,
    )
    trainable_uv: Optional[int] = field(
        default=0,
    )
    nonlin: Optional[int] = field(
        default=0,
    )
    init_type: Optional[int] = field(
        default=1,
    )
    d_init_type: Optional[int] = field(
        default=92,
    )
    custom_scaling: Optional[int] = field(
        default=0,
    )
    use_preconditioner: bool = field(
        default=False,
        metadata={"help": "Enable analytic preconditioners for LoRTA adapters."},
    )
    preconditioner_epsilon: float = field(
        default=1e-6,
        metadata={
            "help": "Stabilization term added to the LoRTA preconditioner before inversion.",
        },
    )
    lora_alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "LoRA alpha"},
    )
    shared_dim: Optional[int] = field(
        default=768,
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default="houlsby",
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )


class CustomTrainer(Trainer):
    def __init__(
        self,
        gradientLogger: GradientLogger,
        classifier_wd,
        classifier_lr,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gradientLogger = gradientLogger
        self.classifier_wd = classifier_wd
        self.classifier_lr = classifier_lr

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: int= None) -> torch.Tensor:
        result = super().training_step(model, inputs, num_items_in_batch)
        self.gradientLogger.log(self.state)
        return result

    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            non_norm_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            classifier_parameters = [
                name for name in non_norm_parameters if "classifier" in name and "bias" not in name
            ]
            classifier_bias_parameters = [
                name for name in non_norm_parameters if "classifier" in name and "bias" in name
            ]
            decay_parameters = [
                name for name in non_norm_parameters if "classifier" not in name and "bias" not in name
            ]
            # for n, p in opt_model.named_parameters():
            #     if p.requires_grad:
            #         if n in classifier_parameters:
            #             print("classifier_parameters", n)
            #         elif n in classifier_bias_parameters:
            #             print("classifier_bias_parameters", n)
            #         elif n in decay_parameters:
            #             print("decay_parameters", n)
            #         else:
            #             print("other_parameters", n)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n not in decay_parameters
                            and n not in classifier_parameters
                            and n not in classifier_bias_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in classifier_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.classifier_wd,
                    "lr": self.classifier_lr,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in classifier_bias_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.classifier_lr,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    job_id = os.environ.get("SLURM_JOB_ID", "0")
    run_id = wandb.util.generate_id()

    print("run_name:", model_args.wandb_run_name)

    is_first_rank = True

    if is_first_rank:
        wandb.init(
            id=run_id,
            group=model_args.wandb_run_group,
            project="",
            name=None if model_args.wandb_run_name is None else model_args.wandb_run_name,
            mode="online" if model_args.wandb_offline == 0 else "offline",
        )
        wandb.config.update(model_args.__dict__)

    # torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
    # logger.info(
    #     "use_deterministic_algorithms: "
    #     + str(torch.are_deterministic_algorithms_enabled())
    # )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        cls_dropout=None,
        apply_lora=model_args.apply_lora,
        lora_alpha=model_args.lora_alpha,
        lora_r=model_args.lora_r,
        apply_adapter=model_args.apply_adapter,
        adapter_type=model_args.adapter_type,
        adapter_size=model_args.adapter_size,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    print(model)

    target_modules = ["query_proj", "value_proj"] if "deberta" in model_args.model_name_or_path else ["query", "value"]
    if model_args.finetune_classifier == 1:
        target_modules += ["classifier.dense", "classifier.out_proj"]

    head_only = model_args.mode == "none"
    if head_only:
        target_modules = ["classifier.dense", "classifier.out_proj"]

    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["query", "key", "value", "attention.output.dense"],  # target_modules,
        lora_dropout=0.0,
        bias="none",
        modules_to_save=[] if model_args.finetune_classifier == 1 else ["classifier"],
        task_type=TaskType.SEQ_CLS,
        use_preconditioner=model_args.use_preconditioner,
        preconditioner_epsilon=model_args.preconditioner_epsilon,
    )
    config.custom = {
        "mode": "lora" if head_only else model_args.mode,
        "submode": model_args.submode,
        "d_init": 1.0,
        "sqrt_a": 5.0,
        "identity": model_args.identity_init == 1,
        "init_type": model_args.init_type,
        "d_init_type": model_args.d_init_type,
        "custom_scaling": model_args.custom_scaling,
        "shared_dim": {"A": model_args.shared_dim, "B": model_args.shared_dim} if model_args.shared_uv == 1 else None,
        "shared_matrices": None,
        "shared_d": model_args.shared_d == 1,
        "shared_d_vector": None,
        "trainable_uv": model_args.trainable_uv == 1,
        "nonlin": model_args.nonlin,
        "use_float64": model_args.use_float64 == 1,
        "norm_penalty": model_args.norm_penalty,
        "norm_alpha": model_args.norm_alpha,
    }
    model = get_peft_model(model, config)

    if model_args.finetune_classifier == 1:
        for n, p in model.named_parameters():
            disable = [
                "dense.bias",
                "dense.weight",
                "out_proj.bias",
                "out_proj.weight",
                "dense.lora_A.",
                "dense.lora_B.",
                "out_proj.lora_A.",
                "out_proj.lora_B.",
            ]
            if any(disable):
                p.requires_grad = False
                print("disabled:", n)
    if head_only:
        for n, p in model.named_parameters():
            disable = ["lora"]
            if any(disable):
                p.requires_grad = False
                print("disabled:", n)

    # model.model.classifier.modules_to_save = None
    print_trainable_parameters(model)

    def get_parameters_count(model, requires_grad=False):
        total_params = 0
        unique_tensors = set()
        print("trainable_named_params:" if requires_grad else "named_params:")
        for name, module in model.named_modules():
            for attr_str in dir(module):
                if attr_str == "trainer":  # Skip the trainer attribute
                    continue
                else:
                    target_attr = getattr(module, attr_str, None)
                    if type(target_attr) in (torch.Tensor, torch.nn.Parameter):
                        if id(target_attr) not in unique_tensors:  # Check if the tensor was already counted
                            if ("classifier" not in name or model_args.finetune_classifier == 1) and (
                                not requires_grad or target_attr.requires_grad
                            ):
                                # print(name, attr_str, target_attr.shape)
                                total_params += torch.numel(target_attr)
                            unique_tensors.add(id(target_attr))  # Add the tensor id to the set of counted tensors

        return total_params

    # trainable_params = []
    # if model_args.apply_lora:
    #     if model_args.lora_path is not None:
    #         lora_state_dict = torch.load(model_args.lora_path)
    #         logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
    #         logger.info(lora_state_dict.keys())
    #         model.load_state_dict(lora_state_dict, strict=False)
    #     trainable_params.append("lora")

    # if model_args.apply_adapter:
    #     if model_args.adapter_path is not None:
    #         adapter_state_dict = torch.load(
    #             os.path.join(model_args.adapter_path, "pytorch_adapter.bin")
    #         )
    #         head_state_dict = torch.load(
    #             os.path.join(model_args.adapter_path, "pytorch_model_head.bin")
    #         )
    #         added_state_dict = {}
    #         for k, v in adapter_state_dict.items():
    #             new_k = (
    #                 k.replace(data_args.task_name + ".", "")
    #                 .replace("adapter_down.0.", "adapter_A.")
    #                 .replace("adapter_up.", "adapter_B.")
    #                 .replace(".adapters.", ".adapter.")
    #             )
    #             added_state_dict[new_k] = v
    #         for k, v in head_state_dict.items():
    #             new_k = k.replace(
    #                 "heads." + data_args.task_name + ".1", "classifier.dense"
    #             ).replace("heads." + data_args.task_name + ".4", "classifier.out_proj")
    #             added_state_dict[new_k] = v
    #         logger.info(f"Apply adapter state dict from {model_args.adapter_path}.")
    #         logger.info(added_state_dict.keys())
    #         missing_keys, unexpected_keys = model.load_state_dict(
    #             added_state_dict, strict=False
    #         )
    #         for missing_key in missing_keys:
    #             assert "adapter" not in missing_key, (
    #                 missing_key + " is missed in the model"
    #             )
    #         assert len(unexpected_keys) == 0, "Unexpected keys " + str(unexpected_keys)
    #     trainable_params.append("adapter")

    # if model_args.apply_bitfit:
    #     trainable_params.append("bias")

    # if len(trainable_params) > 0:
    #     for name, param in model.named_parameters():
    #         if name.startswith("deberta") or name.startswith("roberta"):
    #             param.requires_grad = False
    #             for trainable_param in trainable_params:
    #                 if trainable_param in name:
    #                     param.requires_grad = True
    #                     break
    #         else:
    #             param.requires_grad = True

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[lab] if lab != -1 else -1) for lab in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        try:
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        except Exception as e:
            print(e)
            return {}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # todo: to delete
    if model_args.wandb_run_group is None:
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = 100
    training_args.save_strategy = "no"

    gradientLogger = GradientLogger(model=model)
    weightLogCallback = WeightLogCallback(model=model)

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[weightLogCallback],
        gradientLogger=gradientLogger,
        classifier_lr=model_args.classifier_lr,
        classifier_wd=model_args.classifier_wd,
    )
    trainer.create_optimizer()
    # print(trainer.optimizer)

    params_trainable = get_parameters_count(model, requires_grad=True)
    params_total = get_parameters_count(model, requires_grad=False)

    print(f"Trainable parameters: {params_trainable}")
    print(f"Total number of parameters: {params_total}")
    if is_first_rank:
        wandb.config.update({"params_trainable": params_trainable, "params_total": params_total})
        wandb.log({"params_trainable": params_trainable, "params_total": params_total})

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        weights = {}
        for n, p in model.named_parameters():
            if ".lora_":
                weights[n] = p.detach().cpu().numpy()
        if save_full_weights:
            with open(
                os.path.join(
                    "weights",
                    f"{job_id}_{run_id}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(weights, f)
        if push_weights:
            wandb.save(
                os.path.join(
                    "output/model",
                    "adapter_model.safetensors",
                )
            )

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        try:
            if is_first_rank:
                # path: "runs/[timestamp]-[run_id]"
                path = os.path.join(
                    "runs",
                    f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-{run_id}",
                )
                os.makedirs(path, exist_ok=True)
                state_path = os.path.join(path, "trainer_state.json")
                args_path = os.path.join(path, "trainer_args.json")
                model_args_path = os.path.join(path, "model_args.json")
                trainer.state.save_to_json(state_path)
                with open(args_path, "w") as f:
                    json.dump(trainer.args.to_dict(), f)
                with open(model_args_path, "w") as f:
                    json.dump(model_args.__dict__, f)
        except Exception as e:
            print("error", e)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
