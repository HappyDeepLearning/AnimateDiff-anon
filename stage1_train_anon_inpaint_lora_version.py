import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
# import bitsandbytes as bnb
from tqdm.auto import tqdm

from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

import bitsandbytes as bnb

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.data.anon_dataset_stage1 import GaitLUDatasetImg
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print
import pickle
from PIL import Image

def load_pkl_from_folder(pkl_path, sample_size=256, inference_number=16, is_sil=False):
    if is_sil:
        img_mode = 'L'
    else:
        img_mode = 'RGB'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return_images = [ Image.fromarray(tmp).convert(img_mode).resize((sample_size, sample_size)) for tmp in data]
    if len(return_images) < inference_number:
        return_images = return_images + [return_images[-1]] * (inference_number - len(return_images))
    else:
        return_images = return_images[:inference_number]
    return return_images

def load_images_from_folder(folder, sample_size=256, inference_number=16, is_sil=False):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    # def frame_number(filename):
    #     parts = filename.split('_')
    #     if len(parts) > 1 and parts[0] == 'frame':
    #         try:
    #             return int(parts[1].split('.')[0])  # Extracting the number part
    #         except ValueError:
    #             return float('inf')  # In case of non-integer part, place this file at the end
    #     return float('inf')  # Non-frame files are placed at the end

    # Sorting files based on frame number

    if is_sil:
        img_mode = 'L'
    else:
        img_mode = 'RGB'

    sorted_files = sorted(os.listdir(folder))

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert(img_mode).resize((sample_size, sample_size))
            images.append(img)

    if len(images) < inference_number:
        images = images + [images[-1]] * (inference_number - len(images))
    else:
        images = images[:inference_number]
    return images

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank

def save_pipeline_checkpoint(pipeline, output_dir):
    save_path = os.path.join(output_dir, "pipeline")
    os.makedirs(save_path, exist_ok=True)
    pipeline.save_pretrained(save_path)

def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    is_v100: bool = False,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
    rank: int = 4,

    use_8bit_adam: bool = True,

):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet_inpaint")
        
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set unet trainable parameters
    unet.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)


    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

        new_state_dict = {}
        for key, value in state_dict.items():
            tmp_split = key.split(".")
            new_key = ".".join(tmp_split[1:])
            new_state_dict[new_key] = value

        m, u = unet.load_state_dict(new_state_dict, strict=True)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # for name, param in unet.named_parameters():
    #     for trainable_module_name in trainable_modules:
    #         if trainable_module_name in name:
    #             param.requires_grad = True
    #             break
            
    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())
    trainable_params_list = list(trainable_params)

    if use_8bit_adam:
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params_list,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params_list)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params_list) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = GaitLUDatasetImg(**train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
        validation_pipeline.enable_vae_slicing()
    else:
        validation_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
        validation_pipeline.vae.enable_slicing()

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                mask_values, fg_values = batch["sil_pixel_values"].cpu(), batch["fg_pixel_values"].cpu()
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    mask_values = rearrange(mask_values, "b f c h w -> b c f h w")
                    fg_values = rearrange(fg_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text, mask_value, fg_value) in enumerate(zip(pixel_values, texts, mask_values, fg_values)):
                        pixel_value = pixel_value[None, ...]
                        mask_value = mask_value[None, ...]
                        fg_value = fg_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}_rgb.gif", rescale=True)
                        save_videos_grid(mask_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}_mask.gif", rescale=False)
                        save_videos_grid(fg_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}_fg.gif", rescale=True)
                else:   
                    for idx, (pixel_value, text, mask_value, fg_value) in enumerate(zip(pixel_values, texts, mask_values, fg_values)):
                        pixel_value = pixel_value / 2. + 0.5
                        fg_value = fg_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}_rgb.png")
                        torchvision.utils.save_image(mask_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}_mask.png")
                        torchvision.utils.save_image(fg_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}_fg.png")
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            fg_values = batch["fg_pixel_values"].to(local_rank)
            mask_values = batch["sil_pixel_values"].to(local_rank)

            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

                    fg_values = rearrange(fg_values, "b f c h w -> (b f) c h w")
                    fg_latents = vae.encode(fg_values).latent_dist
                    fg_latents = fg_latents.sample()
                    fg_latents = rearrange(fg_latents, "(b f) c h w -> b c f h w", f=video_length)

                    mask_values = rearrange(mask_values, "b f c h w -> (b f) c h w")
                    h, w = mask_values.shape[-2:]
                    h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
                    scale_factor = 8
                    mask_latents = F.interpolate(mask_values, (h // scale_factor, w // scale_factor))
                    mask_latents = rearrange(mask_latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                    fg_latents = vae.encode(fg_values).latent_dist
                    fg_latents = fg_latents.sample()

                    h, w = mask_values.shape[-2:]
                    h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
                    scale_factor = 8
                    mask_latents = F.interpolate(mask_values, (h // scale_factor, w // scale_factor))

                latents = latents * 0.18215
                fg_latents = fg_latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


            input_latents = torch.cat([noisy_latents, mask_latents, fg_latents], dim=1) # (b, c=4+1+4, f=0 for image, h, w)

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(input_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.state_dict()
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts

                if not image_finetune:
                    inference_number = 16
                else:
                    inference_number = 1

                val_mask_values = load_images_from_folder(validation_data.mask_path, sample_size=train_data.sample_size, inference_number=inference_number, is_sil=True)
                val_image_values = load_images_from_folder(validation_data.image_path, sample_size=train_data.sample_size, inference_number=inference_number, is_sil=False)


                if not image_finetune:
                    validation_data["image"] = val_image_values
                    validation_data["mask_image"] = val_mask_values

                for idx, prompt in enumerate(prompts):
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.sample_n_frames,
                            height       = height,
                            width        = width,
                            **validation_data,
                        ).videos
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)
                        
                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                            guidance_scale      = validation_data.get("guidance_scale", 8.),
                            mask_image          = val_mask_values[0],
                            image               = val_image_values[0],
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        mask_sample = torchvision.transforms.functional.to_tensor(val_mask_values[0]).repeat(3, 1, 1)
                        sample = torch.cat([sample, mask_sample], dim=-1) # C H 2W
                        samples.append(sample)
                
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                if is_main_process:
                    save_pipeline_checkpoint(validation_pipeline, output_dir)
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")

    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, use_8bit_adam=args.use_8bit_adam, **config)
