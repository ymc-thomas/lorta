# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import math
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from functools import partial
from itertools import chain
from typing import Any, Literal, Optional

import torch
from torch import nn

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORTA_PREFIX_MAPPING,
    TRANSFORMERS_MODELS_TO_LORTA_QKVO_MAPPING,
    TRANSFORMERS_MODELS_TO_LORTA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
    get_quantization_config,
)

from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LorTaConfig
from .gptq import dispatch_gptq
from .layer import Linear as LorTaLinear
from .layer import LorTaLayer, dispatch_default
from .tp_layer import dispatch_megatron


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


class LorTaModel(BaseTuner):
    """
    Creates Low Rank Tensor Adapter (LoRTA) model from a pretrained transformers model.

    The method is described in detail in [Coming Soon].

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LorTaConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LoraModel, LorTaConfig

        >>> config = LorTaConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import torch
        >>> import transformers
        >>> from peft import LorTaConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        >>> rank = ...
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LorTaConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )
        >>> quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

        >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     bos_token="[BOS]",
        ...     eos_token="[EOS]",
        ...     unk_token="[UNK]",
        ...     pad_token="[PAD]",
        ...     mask_token="[MASK]",
        ... )
        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     quantization_config=quantization_config,
        ... )
        >>> model = prepare_model_for_kbit_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LorTaConfig`]): The configuration of the Lora model.
    """

    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name) -> None:
        # NOTE: BaseTuner.__init__ calls `inject_adapter` which relies on the
        # `_preconditioner_handles` container being present. Initialise it
        # up-front so that hook setup during adapter injection is safe even on
        # repeated adapter loads.
        self._preconditioner_handles = {}
        super().__init__(model, config, adapter_name)
        
    def _map_layer_to_adapter(self, layer_idx: int, target_matrix: str) -> str:
        return ".".join([self.target_names_prefix, f"{layer_idx}", self.qkvo_mapping[target_matrix]])

    def _compute_weights_from_tensor(self):
        weights = {}
        # print("*"*100)
        # print(self.adapter_name_to_module)
        # print("*"*100)
        for block_idx in range(self.model.config.num_hidden_layers):
            weights[self._map_layer_to_adapter(block_idx, "q")] = torch.cat(
                [
                    self.model.lora_A
                    @ torch.diag(
                        self.model.lora_C_h[head_idx] * self.model.lora_C_m[0] * self.model.lora_C_l[block_idx]
                    )
                    @ self.model.lora_B
                    for head_idx in range(self.model.config.num_attention_heads)
                ],
                dim=1,
            )
            weights[self._map_layer_to_adapter(block_idx, "k")] = torch.cat(
                [
                    self.model.lora_A
                    @ torch.diag(
                        self.model.lora_C_h[head_idx] * self.model.lora_C_m[1] * self.model.lora_C_l[block_idx]
                    )
                    @ self.model.lora_B
                    for head_idx in range(self.model.config.num_attention_heads)
                ],
                dim=1,
            )
            weights[self._map_layer_to_adapter(block_idx, "v")] = torch.cat(
                [
                    self.model.lora_A
                    @ torch.diag(
                        self.model.lora_C_h[head_idx] * self.model.lora_C_m[2] * self.model.lora_C_l[block_idx]
                    )
                    @ self.model.lora_B
                    for head_idx in range(self.model.config.num_attention_heads)
                ],
                dim=1,
            )
            weights[self._map_layer_to_adapter(block_idx, "o")] = torch.cat(
                [
                    self.model.lora_A
                    @ torch.diag(
                        self.model.lora_C_h[head_idx] * self.model.lora_C_m[3] * self.model.lora_C_l[block_idx]
                    )
                    @ self.model.lora_B
                    for head_idx in range(self.model.config.num_attention_heads)
                ],
                dim=1,
            )
        return weights

    def forward(self, *args, **kwargs):
        self.tensor_weights = self._compute_weights_from_tensor()
        # print("*"*100)
        # print(self.tensor_weights.keys())
        # kwargs["adapter_weight"] = weights
        return self.model.forward(*args, **kwargs)

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
        """
        peft_config = self.peft_config[adapter_name]
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        _has_modules_to_save = False

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        self.target_names_prefix = TRANSFORMERS_MODELS_TO_LORTA_PREFIX_MAPPING.get(model_config["model_type"], None)
        self.qkvo_mapping = TRANSFORMERS_MODELS_TO_LORTA_QKVO_MAPPING.get(model_config["model_type"], None)

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        self._prepare_model(peft_config, model)
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        # update peft_config.target_modules if required
        # peft_config = _maybe_include_all_linear_layers(peft_config, model)

        # Create tensor parameters for the adapter
        # print("*"*100)
        # print("Inject Adapter")
        # print(key_list)
        for key in key_list:
            # Check for modules_to_save in case
            if _check_for_modules_to_save and any(
                key.endswith(f"{module_to_save}") for module_to_save in peft_config.modules_to_save
            ):
                # Optionally set the modules to save
                parent, target, target_name = _get_submodules(model, key)

                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)

                _has_modules_to_save = True
                # print("*"*100)
                # print(target_name)
                # print(new_module)
                # print(key)
                # print("Modules to SAVE")
                continue

            if not self._check_target_module_exists(peft_config, key):
                # print("*"*100)
                # print(key)
                # print("Not in target modules")
                continue

            self.targeted_module_names.append(key)
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)
            # print(key)
            # print(adapter_name)
            # print(target)
            # print(target_name)
            # print(parent)
            # print("*"*100)
            # print(new_module)
            # assert(0)
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        if _has_modules_to_save:
            if not hasattr(model, "modules_to_save"):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        A_shape = (self.model.config.hidden_size, peft_config.r)
        B_shape = (peft_config.r, head_dim)
        C_shape = (peft_config.r,)
        self.model.lora_C_l = nn.Parameter(torch.zeros((self.model.config.num_hidden_layers,) + C_shape))
        nn.init.kaiming_uniform_(self.model.lora_C_l, a=math.sqrt(5) * peft_config.init_scale)
        self.model.lora_C_h = nn.Parameter(torch.zeros((self.model.config.num_attention_heads,) + C_shape))
        nn.init.kaiming_uniform_(self.model.lora_C_h, a=math.sqrt(5) * peft_config.init_scale)
        self.model.lora_C_m = nn.Parameter(torch.zeros((4,) + C_shape))
        nn.init.kaiming_uniform_(self.model.lora_C_m, a=math.sqrt(5) * peft_config.init_scale)

        self.model.lora_A = nn.Parameter(torch.empty(A_shape))
        self.model.lora_B = nn.Parameter(torch.zeros(B_shape))
        if len(A_shape) == 2:
            nn.init.kaiming_uniform_(self.model.lora_A, a=math.sqrt(5))
        elif len(A_shape) == 3:
            for i in range(A_shape[0]):
                nn.init.kaiming_uniform_(self.model.lora_A[i], a=math.sqrt(5))
        elif len(A_shape) == 4:
            for i in range(A_shape[0]):
                for j in range(A_shape[1]):
                    nn.init.kaiming_uniform_(self.model.lora_A[i, j], a=math.sqrt(5))

        self._mark_only_adapters_as_trainable(model)
        self._setup_preconditioner_hooks()

    def weights_pre_forward_hook(self, target, args, kwargs, module_name):
        # pre-forward hook to inject weights
        # print(self.tensor_weights.keys())
        kwargs["adapter_weight"] = self.tensor_weights[module_name]
        return args, kwargs

    def _remove_preconditioner_hooks(self):
        handles = getattr(self, "_preconditioner_handles", None)
        if not handles:
            return

        for module_handles in handles.values():
            for handle in module_handles:
                handle.remove()

        handles.clear()

    def _setup_preconditioner_hooks(self):
        handles = getattr(self, "_preconditioner_handles", None)
        if handles is None:
            self._preconditioner_handles = {}
            handles = self._preconditioner_handles

        # Ensure we do not leak handles when loading adapters repeatedly.
        self._remove_preconditioner_hooks()

        for name, module in self.model.named_modules():
            if isinstance(module, (LorTaLayer, LorTaLinear)):
                pre_forward = partial(self.weights_pre_forward_hook, module_name=name)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                handles.setdefault(name, []).append(handle)

    def _check_new_adapter_config(self, config: LorTaConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        return
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _prepare_model(self, peft_config: LorTaConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        if peft_config.layer_replication:  # (TODO: IH - wtf is this ? it clones al OG layers, but with what purpose?)
            replicate_layers(model, peft_config.layer_replication)
        # save all adapter names
        self.adapter_name_to_module = {}
        self.adapter_module_to_name = {}
        # print("Prepare")
        for name, module in self.model.named_modules():
            self.adapter_name_to_module[name] = module
            self.adapter_module_to_name[module] = name
        # print("Prepare done")
        # print(self.adapter_name_to_module.keys())

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LorTaLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            # print("*"*20)
            # print("_create_and_replace")
            # print(target_name, target)
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            # print(new_module)
            # old_name = self.adapter_module_to_name.pop(target)
            # self.adapter_module_to_name[new_module] = old_name
            # self.adapter_name_to_module[old_name] = new_module
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = child.qweight if hasattr(child, "qweight") else child.weight
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False
        self.model.lora_A.requires_grad = True
        self.model.lora_B.requires_grad = True
        self.model.lora_C_l.requires_grad = True
        self.model.lora_C_h.requires_grad = True
        self.model.lora_C_m.requires_grad = True

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, LorTaLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")
    
    def _get_preconditioner_config(self) -> Optional[LorTaConfig]:
        for adapter_name, config in self.peft_config.items():
            if getattr(config, "use_preconditioner", False):
                return config
        return None
    
    def _setup_preconditioner_hooks(self) -> None:
        config = self._get_preconditioner_config()
        if config is None:
            return
        epsilon = float(getattr(config, "preconditioner_epsilon", 1e-6))

        if "lora_A" not in self._preconditioner_handles:
            def _hook_A(grad: torch.Tensor) -> torch.Tensor:
                return self._apply_A_preconditioner(grad, epsilon)

            handle_A = self.model.lora_A.register_hook(_hook_A)
            self._preconditioner_handles["lora_A"] = handle_A

        if "lora_B" not in self._preconditioner_handles:
            def _hook_B(grad: torch.Tensor) -> torch.Tensor:
                return self._apply_B_preconditioner(grad, epsilon)

            handle_B = self.model.lora_B.register_hook(_hook_B)
            self._preconditioner_handles["lora_B"] = handle_B

        if "lora_C_h" not in self._preconditioner_handles:
            def _hook_C_h(grad: torch.Tensor) -> torch.Tensor:
                return self._apply_C_h_preconditioner(grad, epsilon)

            handle_C_h = self.model.lora_C_h.register_hook(_hook_C_h)
            self._preconditioner_handles["lora_C_h"] = handle_C_h

        if "lora_C_l" not in self._preconditioner_handles:
            def _hook_C_l(grad: torch.Tensor) -> torch.Tensor:
                return self._apply_C_l_preconditioner(grad, epsilon)

            handle_C_l = self.model.lora_C_l.register_hook(_hook_C_l)
            self._preconditioner_handles["lora_C_l"] = handle_C_l

        if "lora_C_m" not in self._preconditioner_handles:
            def _hook_C_m(grad: torch.Tensor) -> torch.Tensor:
                return self._apply_C_m_preconditioner(grad, epsilon)

            handle_C_m = self.model.lora_C_m.register_hook(_hook_C_m)
            self._preconditioner_handles["lora_C_m"] = handle_C_m

    def _apply_A_preconditioner(self, grad: Optional[torch.Tensor], epsilon: float) -> Optional[torch.Tensor]:
        if grad is None:
            return grad

        preconditioner = self._compute_A_preconditioner_matrix(epsilon)
        preconditioner = preconditioner.to(device=grad.device, dtype=grad.dtype)
        return grad @ preconditioner
    
    def _apply_B_preconditioner(self, grad: Optional[torch.Tensor], epsilon: float) -> Optional[torch.Tensor]:
        if grad is None:
            return grad

        preconditioner = self._compute_B_preconditioner_matrix(epsilon)
        preconditioner = preconditioner.to(device=grad.device, dtype=grad.dtype)
        return preconditioner @ grad
    
    def _apply_C_h_preconditioner(self, grad: Optional[torch.Tensor], epsilon: float) -> Optional[torch.Tensor]:
        if grad is None:
            return grad

        preconditioner = self._compute_C_h_preconditioner_matrix(epsilon)
        preconditioner = preconditioner.to(device=grad.device, dtype=grad.dtype)
        return grad @ preconditioner
    
    def _apply_C_l_preconditioner(self, grad: Optional[torch.Tensor], epsilon: float) -> Optional[torch.Tensor]:
        if grad is None:
            return grad

        preconditioner = self._compute_C_l_preconditioner_matrix(epsilon)
        preconditioner = preconditioner.to(device=grad.device, dtype=grad.dtype)
        return grad @ preconditioner
    
    def _apply_C_m_preconditioner(self, grad: Optional[torch.Tensor], epsilon: float) -> Optional[torch.Tensor]:
        if grad is None:
            return grad

        preconditioner = self._compute_C_m_preconditioner_matrix(epsilon)
        preconditioner = preconditioner.to(device=grad.device, dtype=grad.dtype)
        return grad @ preconditioner

    def _compute_A_preconditioner_matrix(self, epsilon: float) -> torch.Tensor:
        with torch.no_grad():
            B = self.model.lora_B.detach()
            C_l = self.model.lora_C_l.detach()
            C_h = self.model.lora_C_h.detach()
            C_m = self.model.lora_C_m.detach()

            dtype = B.dtype
            device = B.device

            BBT = B @ B.transpose(0, 1)

            # Build every combination of diagonal factors so we can accumulate the
            # Σ_{h,l,m} diag(C_h[h]) diag(C_l[l]) diag(C_m[m]) B Bᵀ diag(C_m[m]) diag(C_l[l]) diag(C_h[h])
            # term exactly as derived.
            preconditioner_matrix = torch.zeros_like(BBT, device=device, dtype=dtype)

            for l in range(C_l.size(0)):
                for h in range(C_h.size(0)):
                    for m in range(C_m.size(0)):
                        d = C_h[h] * C_l[l] * C_m[m]
                        D = torch.diag(d)
                        preconditioner_matrix += D @ BBT @ D.T

            identity = torch.eye(preconditioner_matrix.size(0), device=device, dtype=dtype) * epsilon
            stabilized = preconditioner_matrix + identity

            try:
                preconditioner = torch.linalg.inv(stabilized)
            except RuntimeError:
                preconditioner = torch.linalg.pinv(stabilized)

            if not torch.isfinite(preconditioner).all():
                preconditioner = torch.linalg.pinv(stabilized)

            return preconditioner

    def _compute_B_preconditioner_matrix(self, epsilon: float) -> torch.Tensor:
        with torch.no_grad():
            A = self.model.lora_A.detach()
            C_l = self.model.lora_C_l.detach()
            C_h = self.model.lora_C_h.detach()
            C_m = self.model.lora_C_m.detach()

            dtype = A.dtype
            device = A.device

            ATA = A.transpose(0, 1) @ A

            preconditioner_matrix = torch.zeros_like(ATA, device=device, dtype=dtype)

            for l in range(C_l.size(0)):
                for h in range(C_h.size(0)):
                    for m in range(C_m.size(0)):
                        d = C_h[h] * C_l[l] * C_m[m]
                        D = torch.diag(d)
                        preconditioner_matrix += D.transpose(0, 1) @ ATA @ D

            identity = torch.eye(preconditioner_matrix.size(0), device=device, dtype=dtype) * epsilon
            stabilized = preconditioner_matrix + identity

            try:
                preconditioner = torch.linalg.inv(stabilized)
            except RuntimeError:
                preconditioner = torch.linalg.pinv(stabilized)

            if not torch.isfinite(preconditioner).all():
                preconditioner = torch.linalg.pinv(stabilized)

            return preconditioner  
        
    def _compute_C_h_preconditioner_matrix(self, epsilon: float) -> torch.Tensor:
        with torch.no_grad():
            A = self.model.lora_A.detach()
            B = self.model.lora_B.detach()
            C_l = self.model.lora_C_l.detach()
            C_m = self.model.lora_C_m.detach()

            dtype = A.dtype
            device = A.device

            ATA = A.transpose(0, 1) @ A
            BBT = B @ B.transpose(0, 1)

            preconditioner_matrix = torch.zeros_like(ATA, device=device, dtype=dtype)

            for l in range(C_l.size(0)):
                for m in range(C_m.size(0)):
                    d = C_l[l] * C_m[m]
                    D = torch.diag(d)
                    preconditioner_matrix += ATA * (D @ BBT @ D.transpose(0, 1))

            identity = torch.eye(preconditioner_matrix.size(0), device=device, dtype=dtype) * epsilon
            stabilized = preconditioner_matrix + identity

            try:
                preconditioner = torch.linalg.inv(stabilized)
            except RuntimeError:
                preconditioner = torch.linalg.pinv(stabilized)

            if not torch.isfinite(preconditioner).all():
                preconditioner = torch.linalg.pinv(stabilized)

            return preconditioner.transpose(0,1)
        
    def _compute_C_l_preconditioner_matrix(self, epsilon: float) -> torch.Tensor:
        with torch.no_grad():
            A = self.model.lora_A.detach()
            B = self.model.lora_B.detach()
            C_h = self.model.lora_C_h.detach()
            C_m = self.model.lora_C_m.detach()

            dtype = A.dtype
            device = A.device

            ATA = A.transpose(0, 1) @ A
            BBT = B @ B.transpose(0, 1)

            preconditioner_matrix = torch.zeros_like(ATA, device=device, dtype=dtype)

            for h in range(C_h.size(0)):
                for m in range(C_m.size(0)):
                    d = C_h[h] * C_m[m]
                    D = torch.diag(d)
                    preconditioner_matrix += (D.transpose(0, 1) @ ATA @ D) * BBT

            identity = torch.eye(preconditioner_matrix.size(0), device=device, dtype=dtype) * epsilon
            stabilized = preconditioner_matrix + identity

            try:
                preconditioner = torch.linalg.inv(stabilized)
            except RuntimeError:
                preconditioner = torch.linalg.pinv(stabilized)

            if not torch.isfinite(preconditioner).all():
                preconditioner = torch.linalg.pinv(stabilized)

            return preconditioner.transpose(0, 1)
        
    def _compute_C_m_preconditioner_matrix(self, epsilon: float) -> torch.Tensor:
        with torch.no_grad():
            A = self.model.lora_A.detach()
            B = self.model.lora_B.detach()
            C_l = self.model.lora_C_l.detach()
            C_h = self.model.lora_C_h.detach()

            dtype = A.dtype
            device = A.device

            ATA = A.transpose(0, 1) @ A
            BBT = B @ B.transpose(0, 1)

            preconditioner_matrix = torch.zeros_like(ATA, device=device, dtype=dtype)

            for l in range(C_l.size(0)):
                D1 = torch.diag(C_l[l])
                for h in range(C_h.size(0)):
                    D2 = torch.diag(C_h[h])
                    left = D2.transpose(0, 1) @ ATA @ D2
                    right = D1 @ BBT @ D1.transpose(0, 1)
                    preconditioner_matrix += left * right

            identity = torch.eye(preconditioner_matrix.size(0), device=device, dtype=dtype) * epsilon
            stabilized = preconditioner_matrix + identity

            try:
                preconditioner = torch.linalg.inv(stabilized)
            except RuntimeError:
                preconditioner = torch.linalg.pinv(stabilized)

            if not torch.isfinite(preconditioner).all():
                preconditioner = torch.linalg.pinv(stabilized)

            return preconditioner.transpose(0, 1)
    
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend([dispatch_aqlm, dispatch_awq, dispatch_gptq, dispatch_megatron, dispatch_default])

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LorTaLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        ##adapter_names = kwargs.pop("adapter_names", None)
        # if adapter_names is None:
        # nothing to do
        # yield
        # return
        # else:
        # raise NotImplementedError
        # if self.training:
        # raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        hook_handles = []
        names_with_hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, LorTaLayer) or isinstance(module, LorTaLinear):
                pre_forward = partial(self.weights_pre_forward_hook, module_name=name)
                # old_module_name = self.adapter_module_to_name[module.base_layer]
                # old_module = self.adapter_name_to_module.pop(old_module_name)
                # self.adapter_name_to_module[old_module_name] = module
                # self.adapter_module_to_name[module] = old_module_name
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)
                names_with_hooks.append(name)
        yield
        # print("*"*100)
        # print("Hooked layers")
        # print(names_with_hooks)

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge LORA layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge LORA layers when base model layers are replicated")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORTA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORTA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        r"""
        Internal method to merge the adapters into the base model and unload them.

        Args:
            merge (`bool`):
                Whether to merge the adapters into the base model's weights.
            progressbar (`bool`):
                Whether to show a progress bar during the process.
            safe_merge (`bool`):
                Whether to perform a safety check for NaNs or Infs in the adapter weights.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names to merge. If `None`, all active adapters are merged.

        Returns:
            `torch.nn.Module`: The merged model.
        """
        self._remove_preconditioner_hooks()
        self._check_merge_allowed()

        if adapter_names is None:
            adapter_names = self.active_adapters

        # Ensure adapter_names is a list
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Collect modules to process
        modules_to_process = []
        for name, module in self.model.named_modules():
            if isinstance(module, (LorTaLayer, LorTaLinear)):
                #if module.active_adapter in adapter_names:
                modules_to_process.append((name, module))

        # Optionally use a progress bar
        if progressbar:
            from tqdm.auto import tqdm
            modules_to_process = tqdm(modules_to_process, desc="Processing adapters", total=len(modules_to_process))

        if merge:
            # Compute the adapter weights
            self.tensor_weights = self._compute_weights_from_tensor()

            # Process modules
            for name, module in modules_to_process:
                # Get the base layer
                base_module = module.base_layer
                # Get the adapter weight
                adapter_weight = self.tensor_weights.get(name, None)
                if adapter_weight is not None:
                    # Perform safe merge check if requested
                    if safe_merge:
                        if torch.isnan(adapter_weight).any() or torch.isinf(adapter_weight).any():
                            raise ValueError(f"NaN or Inf detected in adapter weights for module {name}")

                    # Merge the adapter weight into the base weight
                    base_module.weight.data += adapter_weight.to(base_module.weight.device) * module.scaling["default"]
                # Replace the module with the base_layer
                parent, _, target_name = _get_submodules(self.model, name)
                setattr(parent, target_name, base_module)
        else:
            # If not merging, simply replace the modules with their base layers
            for name, module in modules_to_process:
                base_module = module.base_layer
                parent, _, target_name = _get_submodules(self.model, name)
                setattr(parent, target_name, base_module)

        # Remove the adapter parameters
        for param_name in ['lora_A', 'lora_B', 'lora_C_l', 'lora_C_h', 'lora_C_m']:
            if hasattr(self.model, param_name):
                delattr(self.model, param_name)

        # Set requires_grad=False for all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Remove the adapters from peft_config
        for adapter_name in adapter_names:
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]

        return self.model


    def add_weighted_adapter(
        self,
        adapters,
        weights,
        adapter_name,
        combination_type="svd",
        svd_rank=None,
        svd_clamp=None,
        svd_full_matrices=True,
        svd_driver=None,
        density=None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        raise NotImplementedError

    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        raise NotImplementedError

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        raise NotImplementedError

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        raise NotImplementedError

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRTA adapters into the base model and unloads them. This is useful when you want to use
        the base model as a standalone model without the adapter layers.

        Args:
            progressbar (`bool`):
                Whether to show a progress bar indicating the merge and unload process.
            safe_merge (`bool`):
                Whether to perform a safety check for NaNs or Infs in the adapter weights before merging.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.

        Returns:
            `torch.nn.Module`: The merged model.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import LorTaModel, LorTaConfig

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> lorta_config = LorTaConfig(...)
        >>> model = LorTaModel(base_model, lorta_config)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            merge=True, progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )
        