
import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from main.controlnet.pretrained import get_pretrained_controlnet_model
from stable_audio_tools.inference.sampling import get_alphas_sigmas


class ModelMultiStem(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        depth_factor: float,
        cfg_dropout_prob: float
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.cfg_dropout_prob = cfg_dropout_prob

        self.timestep_sampler = "logit_normal"
        self.diffusion_objective = "v"

        model, model_config = get_pretrained_controlnet_model(
            "stabilityai/stable-audio-open-1.0",
            controlnet_types=["audio"],
            depth_factor=depth_factor
        )

        self.model = model
        self.model_config = model_config
        self.sample_size = model_config["sample_size"]
        self.sample_rate = model_config["sample_rate"]

        self.model.model.model.requires_grad_(False)
        self.model.conditioner.requires_grad_(False)
        self.model.conditioner.eval()
        self.model.pretransform.requires_grad_(False)
        self.model.pretransform.eval()

        # 多轨输出头
        self.output_heads = torch.nn.ModuleDict({
            "bass":   torch.nn.Conv1d(model_config["model_channels"], 1, kernel_size=1),
            "drums":  torch.nn.Conv1d(model_config["model_channels"], 1, kernel_size=1),
            "vocals": torch.nn.Conv1d(model_config["model_channels"], 1, kernel_size=1),
            "other":  torch.nn.Conv1d(model_config["model_channels"], 1, kernel_size=1),
        })

    def configure_optimizers(self):
        params = list(self.model.model.controlnet.parameters()) + list(self.output_heads.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def step(self, batch):
        x, y_dict, prompts, start_seconds, total_seconds = batch

        diffusion_input = self.model.pretransform.encode(x)

        if self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(x.shape[0]))
        else:
            raise ValueError(f"Unknown time step sampler: {self.timestep_sampler}")

        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        else:
            raise ValueError("Diffusion objective not supported")

        alphas = alphas[:, None, None].to(self.device)
        sigmas = sigmas[:, None, None].to(self.device)

        noise = torch.randn_like(diffusion_input).to(self.device)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas

        cond = self.model.conditioner([
            {"prompt": prompts[i],
             "seconds_start": start_seconds[i],
             "seconds_total": total_seconds[i],
             "audio": y_dict["humming"][i:i+1]}
            for i in range(x.shape[0])
        ], device=self.device)

        features = self.model(
            x=noised_inputs,
            t=t.to(self.device),
            cond=cond,
            cfg_dropout_prob=self.cfg_dropout_prob
        )  # [B, C, T]

        output_dict = {k: self.output_heads[k](features) for k in self.output_heads}

        loss = 0.0
        for k in output_dict:
            if k in y_dict:
                loss += F.mse_loss(output_dict[k], y_dict[k])
        loss = loss / len(output_dict)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("valid_loss", loss)
        return loss
