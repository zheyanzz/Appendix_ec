

import copy

import logging

import math


import torch

from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf


from .loss_scheduler import LossScheduler


logger = logging.getLogger(__name__)


class EMAModel:


    def __init__(self, model, decay: float = 0.9999):

        self.decay = decay

        self.shadow = {}

        for name, param in model.named_parameters():

            if param.requires_grad:

                self.shadow[name] = param.data.clone()


    @torch.no_grad()

    def update(self, model):

        for name, param in model.named_parameters():

            if name in self.shadow:

                self.shadow[name].mul_(self.decay).add_(

                    param.data, alpha=1 - self.decay

                )


    def apply(self, model):

        self._backup = {}

        for name, param in model.named_parameters():

            if name in self.shadow:

                self._backup[name] = param.data.clone()

                param.data.copy_(self.shadow[name])


    def restore(self, model):

        for name, param in model.named_parameters():

            if name in self._backup:

                param.data.copy_(self._backup[name])

        self._backup = {}


class Trainer:


    def __init__(self, model, cfg: DictConfig, accelerator=None):

        self.cfg = cfg

        self.device = cfg.get("device", "cuda")


        

        self.grad_accum_steps = cfg.train.get("gradient_accumulation_steps", 4)


        

        if accelerator is not None:

            self.accelerator = accelerator

        else:

            try:

                from accelerate import Accelerator

                self.accelerator = Accelerator(

                    gradient_accumulation_steps=self.grad_accum_steps,

                    mixed_precision="bf16",

                )

            except ImportError:

                logger.warning(

                    "accelerate not installed — falling back to single-GPU training. "

                    "Install with: pip install accelerate"

                )

                self.accelerator = None


        

        self.stage_boundaries = [0, 62500, 125000, 187500, 250000]


        

        stage_overrides = {}

        if hasattr(cfg, "training_stages"):

            for stage_cfg in cfg.training_stages:

                stage_num = stage_cfg.get("stage", 0)

                if hasattr(stage_cfg, "loss_overrides"):

                    stage_overrides[stage_num] = OmegaConf.to_container(

                        stage_cfg.loss_overrides, resolve=True

                    )

        elif hasattr(cfg.train, "loss_overrides"):

            current_stage = cfg.train.get("stage", 1)

            stage_overrides[current_stage] = OmegaConf.to_container(

                cfg.train.loss_overrides, resolve=True

            )

        self.loss_scheduler = LossScheduler(

            base_weights=OmegaConf.to_container(cfg.loss, resolve=True),

            stage_overrides=stage_overrides,

        )


        

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        base_lr = cfg.train.get("lr", 1e-4)

        self.optimizer = torch.optim.AdamW(

            trainable_params,

            lr=base_lr,

            weight_decay=cfg.train.get("weight_decay", 1e-2),

        )


        

        warmup_steps = cfg.train.get("warmup_steps", 1000)

        total_steps = self.stage_boundaries[-1]

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(

            self.optimizer,

            lr_lambda=lambda step: self._lr_lambda(step, warmup_steps, total_steps),

        )


        

        if self.accelerator is not None:

            self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(

                model, self.optimizer, self.lr_scheduler

            )

        else:

            self.model = model


        

        ema_decay = cfg.train.get("ema_decay", 0.9999)

        use_ema = cfg.train.get("use_ema", True)

        self.ema = EMAModel(self.model, decay=ema_decay) if use_ema else None


        self.global_step = 0

        self.current_stage = 1

        self._accum_count = 0


    @staticmethod

    def _lr_lambda(step: int, warmup: int, total: int) -> float:

        if step < warmup:

            return step / max(warmup, 1)

        progress = (step - warmup) / max(total - warmup, 1)

        return 0.5 * (1 + math.cos(math.pi * progress))


    def get_stage(self, step: int) -> int:

        for i in range(len(self.stage_boundaries) - 1):

            if self.stage_boundaries[i] <= step < self.stage_boundaries[i + 1]:

                return i + 1

        return 4


    def train_step(self, batch: dict) -> dict:

        self.current_stage = self.get_stage(self.global_step)

        loss_weights = self.loss_scheduler.get_weights(self.current_stage)


        

        if self.accelerator is not None:

            with self.accelerator.accumulate(self.model):

                with self.accelerator.autocast():

                    outputs = self.model.forward_training(

                        batch, loss_weights, self.global_step

                    )

                total_loss = outputs["total_loss"]

                self.accelerator.backward(total_loss)


                if self.accelerator.sync_gradients:

                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)


                self.optimizer.step()

                self.lr_scheduler.step()

                self.optimizer.zero_grad()


                if self.accelerator.sync_gradients:

                    if self.ema is not None:

                        self.ema.update(self.model)

                    self.global_step += 1

        else:

            

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                outputs = self.model.forward_training(

                    batch, loss_weights, self.global_step

                )


            total_loss = outputs["total_loss"] / self.grad_accum_steps

            total_loss.backward()

            self._accum_count += 1


            if self._accum_count >= self.grad_accum_steps:

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                self.lr_scheduler.step()

                self.optimizer.zero_grad()

                if self.ema is not None:

                    self.ema.update(self.model)

                self._accum_count = 0

                self.global_step += 1


        loss_dict = outputs.get("loss_dict", {})

        loss_dict["stage"] = self.current_stage

        loss_dict["lr"] = self.optimizer.param_groups[0]["lr"]

        loss_dict["step"] = self.global_step


        return loss_dict


    def train(

        self,

        content_loader: DataLoader,

        style_loader: DataLoader,

        total_steps: int | None = None,

        save_fn=None,

        save_every: int = 5000,

    ):

        from ..data.unpaired_sampler import UnpairedSampler


        total = total_steps or self.stage_boundaries[-1]


        

        if self.accelerator is not None:

            content_loader, style_loader = self.accelerator.prepare(

                content_loader, style_loader

            )


        sampler = UnpairedSampler(content_loader, style_loader)

        self.model.train()


        for content_batch, style_batch in sampler:

            if self.global_step >= total:

                break


            if self.accelerator is not None:

                device = self.accelerator.device

            else:

                device = self.device


            batch = {

                "source_video": content_batch.to(device),

                "style_video": style_batch.to(device),

            }


            loss_dict = self.train_step(batch)


            if self.global_step % 100 == 0 and self.global_step > 0:

                is_main = (

                    self.accelerator is None or self.accelerator.is_main_process

                )

                if is_main:

                    logger.info(

                        "Step %d | Stage %d | Loss %.4f | LR %.2e",

                        loss_dict["step"],

                        loss_dict["stage"],

                        loss_dict.get("L_total", 0.0),

                        loss_dict["lr"],

                    )


            

            new_stage = self.get_stage(self.global_step)

            if new_stage != self.current_stage:

                is_main = (

                    self.accelerator is None or self.accelerator.is_main_process

                )

                if is_main:

                    logger.info("=== Transitioning to Stage %d ===", new_stage)


            

            if (

                save_fn is not None

                and self.global_step % save_every == 0

                and self.global_step > 0

            ):

                is_main = (

                    self.accelerator is None or self.accelerator.is_main_process

                )

                if is_main:

                    

                    if self.ema is not None:

                        self.ema.apply(self.model)

                    save_fn(

                        self.model, self.optimizer,

                        self.global_step, self.current_stage,

                    )

                    if self.ema is not None:

                        self.ema.restore(self.model)


        is_main = self.accelerator is None or self.accelerator.is_main_process

        if is_main:

            logger.info("Training complete at step %d", self.global_step)

