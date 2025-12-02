# LoRTA: Low Rank Tensor Adaptation of Large Language Models

This folder contains the code and resources accompanying the paper submission **[LoRTA: Low Rank Tensor Adaptation of Large Language Models](https://arxiv.org/abs/2410.04060)** (under review).

> **This is the preconditioned LoRTA branch.**  
> It extends the original LoRTA implementation with preconditioners for the LoRTA framework, as used in the experiments of the paper.  
> In this branch, preconditioners are implemented for **RoBERTa-base** on the **GLUE** benchmark.  
> The non-preconditioned (vanilla) LoRTA code is available in the other branches of this repository.

## Repository Structure

The repository is organized into four directories, each containing a readme with instructions.

* **`peft/`** : Implementation of LoRTA in [HFs parameter efficient finetuning library.](https://github.com/huggingface/peft)

The remaining three correspond to a different set of the experiments in the manuscript:

* **`instruction_tuning/`**: Instruction tuning experiments. Based on [lightning-GPT](https://github.com/Lightning-AI/litgpt).
* **`dpo/`**: Direct Preference optimization. It is based on [trl](https://huggingface.co/docs/trl/en/index), and needs peft-lorta to be installed.
* **`nlu/`** Contains the peft implementation of LoRA along with all the GLUE benchmark experiments. Based on [VeRA](https://dkopi.github.io/vera/)'s submission code.


