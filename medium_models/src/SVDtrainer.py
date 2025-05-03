########## The following part is copied from Transformers' trainer (3.4.0) and later ported to be compatible with v4.4.2 and to support initialization from linear head probing. ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from loguru import logger
import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))

class SVDTrainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.named_parameters_to_optim = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()


    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)
    
    def random_gaussian_matrix(self, m, n, device, dtype, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)

        random_matrix = torch.randn(m, n, device=device, dtype=dtype)
        return random_matrix
    


    # =========================================== SVD Functions =======================================================================
    

    def new_zo_subspace_perturb_parameters(self, random_seed=None, scaling_factor=1):
             
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for _, param, U, V in self.named_parameters_to_optim:
            if len(U.shape) == 1:
                
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)    
                param.data = param.data + scaling_factor * z * self.args.zo_eps
                
            elif len(U.shape) == 2:
  
                z = torch.normal(mean=0, std=1, size=(U.shape[1], V.shape[0]), device=param.data.device, dtype=param.data.dtype)
                z = (U @ z @ V * math.sqrt(param.data.numel() / z.numel())).view(param.data.shape)
                
                # param.data = param.data + U @ (scaling_factor * z * self.args.zo_eps * math.sqrt(param.data.numel() / z.numel())) @ V
                param.data = param.data + scaling_factor * z * self.args.zo_eps 

    @torch.no_grad()
    def new_zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                # param.grad = None  # Make sure the grad is empty and will not be updated.

        if args.quantization:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    bnb.functional.dequantize_nf4(param.data, quant_state=param.quant_state, out=param.data)                

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # print("loss1 is: ", loss1)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

        # print("loss2 is: ", loss2)
        # print("projected grad is: ", self.projected_grad)
        # No gradient accumulation support
        # assert self.args.gradient_accumulation_steps == 1

        if args.quantization:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    _, param.quant_state =  bnb.functional.quantize_nf4(param.data, out=param.data)

        return loss1

    @torch.no_grad()
    def new_zo_subspace_step(self, model, inputs):
        """
        Estimate gradient by subzero. Return the loss from f(theta + z)
        """
        args = self.args
                
        if self.state.global_step % args.update_interval == 0:
            # print(args.mode)
            self.new_zo_step(model, inputs)
            print("step is: ", self.state.global_step, "new projected grad is: ",self.projected_grad)

        # What parameters to optimize
        # self.named_parameters_to_optim = []
        self.named_parameters_to_optim_new = []
        for name, param in model.named_parameters():
            if param.requires_grad:
   
                # 增加量化
                # if args.quantization:
                    # print("param name is ", name)
                    # print("original weight is: ", param.data)
                    # quant_weight, quant_state =  bnb.functional.quantize_nf4(param.data, out=param.data)
                    # print("quantize the weight to 4-bit: ", param.data)
                    # self.quant_state[name] = quant_state         
                    # dequant_weight = torch.zeros_like(param.data)
                    # result = bnb.functional.dequantize_nf4(param.data, quant_state=quant_state, out=dequant_weight)
                    # print("dequantize the weight to 4-bit: ", dequant_weight)

                if len(torch.squeeze(param.data).shape) == 2:

                    gauss_rank = min(args.gauss_rank, min(param.data.size(0), param.data.size(1)))

                    # print("gauss rank is: ", gauss_rank)
                    # print("param name is ", name)
                    # print("param data size is: ", param.data.shape)

                    if self.state.global_step == 0:
   
                        self.p_state[name] = {'U': torch.zeros(param.data.size(0), gauss_rank), 
                                                'V': torch.zeros(gauss_rank, param.data.size(1))}
                  
                    p_state = self.p_state[name]           
                    
                    if self.state.global_step % args.update_interval == 0:
                        torch.manual_seed(self.zo_random_seed)
                        # Resample z
                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                        dtype=param.data.dtype)

                        param.grad = self.projected_grad * z              

                        if args.mode in ['lora', 'prefix', 'prompt']:
                            # print(args.mode)
                            # print(param.data.shape)
                            # w_shape = reshape_matrix(param.data.numel())
                            # param.data = param.data.reshape(w_shape)
                            # w_shape = param.data.shape
                            # U, V = fast_svd_method_v2(w_shape=w_shape, device=param.device, dtype=param.data.dtype, rank=args.gauss_rank)
                            param_grad = param.grad
                            U, V = get_orthogonal_matrix(weights=param_grad, rank=gauss_rank)
                        else:
                            # U, V = fast_svd_method_v2(w_shape=param.data.shape, device=param.device, dtype=param.data.dtype, rank=args.gauss_rank)
                            param_grad = param.grad
                            U, V = get_orthogonal_matrix(weights=param_grad, rank=gauss_rank)

                        p_state['U'] = U
                        p_state['V'] = V
                        
                    U = p_state['U']
                    V = p_state['V']  
                    # print("U and V shape is", U.shape, V.shape)
                    
                    self.named_parameters_to_optim_new.append((name, param, U, V))
                else:
                    self.named_parameters_to_optim_new.append((name, param, torch.Tensor([1.]), torch.Tensor([1.])))
                    # # TODO avoid init the memory for grad.
                    # param.grad = torch.zeros_like(param.data)
                    # param.grad = None  # Make sure the grad is empty and will not be updated.
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        self.named_parameters_to_optim = self.named_parameters_to_optim_new
        # Sample the random seed for sampling z
        # torch.cuda.empty_cache()
        self.zo_random_seed = np.random.randint(1000000000)

        # print("named parameters to optim is: ",self.named_parameters_to_optim)

        if args.quantization:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    bnb.functional.dequantize_nf4(param.data, quant_state=param.quant_state, out=param.data)
                    # if (name == "model.decoder.layers.1.self_attn.k_proj.weight"):
                    #     print("param", name, "dequantize the weight to 4-bit: ", param.data)

        # First function evaluation
        self.new_zo_subspace_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.new_zo_subspace_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.new_zo_subspace_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.new_zo_subspace_perturb_parameters(scaling_factor=1)

        if args.quantization:
            # print("after zo subspace step, loss1 is: ", loss1)
            # print("after zo subspace step, loss2 is: ", loss2)
            # print("after zo subspace step, project grad is: ", self.projected_grad)
            pass
        
        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    def new_zo_subspace_update(self, model):

        args = self.args
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(self.zo_random_seed)

        if args.quantization:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    bnb.functional.dequantize_nf4(param.data, quant_state=param.quant_state, out=param.data)     

        for name, param, U, V in self.named_parameters_to_optim:

            # 增加
            # if args.quantization:
                # quant_state = self.quant_state[name]
                # bnb.functional.dequantize_nf4(param.data, quant_state=quant_state, out=param.data)
                # print("dequantize the weight to 4-bit: ", param.data)

            # Resample z
            if len(torch.squeeze(param.data).shape) == 2:    

                gauss_rank = min(args.gauss_rank, min(param.data.size(0), param.data.size(1)))

                z0 = torch.normal(mean=0, std=1, size=(gauss_rank, gauss_rank), device=param.data.device, dtype=param.data.dtype)
                # z = U @ z0 @ V * math.sqrt(param.data.numel() / z0.numel())
                # print("z0 shape is",z0.shape)
                z = (U @ z0 @ V * math.sqrt(param.data.numel() / z0.numel())).view(param.data.shape).to(param.data.dtype)

            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                             dtype=param.data.dtype)

            param.grad = self.projected_grad * z  # NOTE this q division does not work for q>1.

            self.optimizer.step()  # will only update grad that is not None.
            # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
            param.grad = None  # avoid further update.
            
        self.update_steps += 1
        if self.update_steps % 1000 == 0:
            print('model update', self.update_steps)
        # self.optimizer.step()
        # print(type(self.optimizer), self.optimizer)
        self.lr_scheduler.step()  # NOTE When we use own optimizer, this will no longer update the lr anymore.
        # self.optimizer.zero_grad()
        # model.zero_grad()
        
        
    # =====================================================================================================================================
        

    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1 
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError
        # print("Sample %d zs" % (noise_sample_time))

        return noise_sample_time

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):
                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight

                # estimate c's (param or grad norm) on epoch 0
                if epoch == 0 and step == 0 and self.args.zo_variant is not None:
                    self.initialize_c(model, inputs)
                elif step == 0 and self.args.zo_variant is not None and self.args.recompute_norms:
                    self.initialize_c(model, inputs)
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                    
                if self.args.SVD_zo:
                    tr_loss_step = self.new_zo_subspace_step(model, inputs)
                    # tr_loss += tr_loss_step
                    if self.args.SVD_optimizer == 'sgd':
                        self.new_zo_subspace_update()
                    # elif self.args.lozo_optimizer == 'sgdm':
                    #     self.lowrank_zo_update_momentum()
                    else:
                        raise ValueError(f"Unsupported optimizer: {self.args.SVD_optimizer}")
                    self.state.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        logs["loss"] = tr_loss_step.item()
                        logs["learning_rate"] = self.args.learning_rate
                        logs["global_step"] = self.state.global_step
                        logs["max_steps"] = self.args.max_steps
                        logs["time"] = int(time.time() - start_time)
                        self.log(logs)
                        logger.info(str(logs))
                # standard, non-ZO optimization
                else:
                    tr_loss += self.training_step(model, inputs)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(optimizer)
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        elif self.args.fp16:
                            norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if self.args.optimizer_variant == 'signgd':
                            for n,p in model.named_parameters():
                                if p.grad is not None:
                                    p.grad = torch.sign(p.grad)

                        if transformers.is_torch_tpu_available():
                            xm.optimizer_step(optimizer)
                        elif self.args.fp16 and _use_native_amp:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            tr_loss_scalar = tr_loss.item()
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            logs["norm"] = norm.item()
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss_scalar = tr_loss_scalar

                            self.log(logs)
                            logger.info(str(logs))

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                # train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
    
