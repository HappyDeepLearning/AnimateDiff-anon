# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import math
from tqdm import tqdm
import random

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.loaders import LoraLoaderMixin
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import retrieve_latents
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from diffusers.image_processor import VaeImageProcessor

from einops import rearrange
from kornia import morphology as morph

from ..models.unet import UNet3DConditionModel
from ..models.controlnet2d import ControlNetModel
# from ..models.sparse_controlnet import SparseTemporalControlNetModel
from .utils import get_tensor_interpolation_method, set_tensor_interpolation_method
from .context import get_context_scheduler


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline, LoraLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[ControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        set_tensor_interpolation_method(False)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents, decoder_consistency=None):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            if decoder_consistency is not None:
                video.append(decoder_consistency(latents[frame_idx:frame_idx+1]))
            else:
                video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    # def decode_latents(self, latents):
    #     video_length = latents.shape[2]
    #     latents = 1 / 0.18215 * latents
    #     latents = rearrange(latents, "b c f h w -> (b f) c h w")
    #     # video = self.vae.decode(latents).sample
    #     video = []
    #     for frame_idx in tqdm(range(latents.shape[0])):
    #         video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
    #     video = torch.cat(video)
    #     video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    #     video = (video / 2 + 0.5).clamp(0, 1)
    #     # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    #     video = video.cpu().float().numpy()
    #     return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, return_image_latents=False):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # if return_image_latents or (latents is None):
        #     image = image.to(device=device, dtype=dtype)

        #     if image.shape[1] == 4:
        #         image_latents = image
        #     else:
        #         image_latents = self._encode_vae_image(image=image, generator=generator)
        #     image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler

        latents = latents * self.scheduler.init_noise_sigma
        # if return_image_latents:
        #     return [latents, image_latents]
        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_mask_latents(
        self, mask, init_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, inference_frame_number, deliate_mask
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        mask = rearrange(mask, "b c f h w -> (b f) c h w").contiguous()
        init_image = rearrange(init_image, "b c f h w -> (b f) c h w").contiguous()


        if deliate_mask:
            kernel_size = random.randint(3,10)
            kernel = torch.ones((kernel_size, kernel_size))
            print(f"dilation mask with kernel size {kernel_size}")
            mask = morph.dilation(mask, kernel.to(mask.device))

        masked_image = init_image * (mask < 0.5)

        mask = torch.nn.functional.interpolate(mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor))
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        mask = rearrange(mask, "(b f) c h w -> b c f h w", f=inference_frame_number)
        masked_image_latents = rearrange(masked_image_latents, "(b f) c h w -> b c f h w", f=inference_frame_number)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def interpolate_latents(self, latents: torch.Tensor, interpolation_factor:int, device ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
                    (latents.shape[0],latents.shape[1],((latents.shape[2]-1) * interpolation_factor)+1, latents.shape[3],latents.shape[4]),
                    device=latents.device,
                    dtype=latents.dtype,
                )

        org_video_length = latents.shape[2]
        rate = [i/interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0,i1 in zip( range( org_video_length ),range( org_video_length )[1:] ):
            v0 = latents[:,:,i0,:,:]
            v1 = latents[:,:,i1,:,:]

            new_latents[:,:,new_index,:,:] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(v0.to(device=device),v1.to(device=device),f)
                new_latents[:,:,new_index,:,:] = v.to(latents.device)
                new_index += 1

        new_latents[:,:,new_index,:,:] = v1
        new_index += 1

        return new_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        # support controlnet
        control_image = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,

        # support inpainting
        mask_image   = None,
        image        = None,
        deliate_mask= False,

        # long video support
        context_schedule="uniform",
        context_frames=24,
        context_stride=3,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        decoder_consistency=None,
        return_image_latents=False,

        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps


        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=None, resize_mode="default"
        )
        init_image = init_image.to(dtype=text_embeddings.dtype)
        inference_frame_number = len(init_image)

        init_image = init_image.squeeze(0).repeat(batch_size, 1, 1, 1, 1) # [B, T, C, H, W]
        init_image = rearrange(init_image, "b t c h w -> b c t h w")

        if control_image is not None:
            control_image = self.control_image_processor.preprocess(
                control_image, height=height, width=width, crops_coords=None, resize_mode="default"
            ).to(dtype=text_embeddings.dtype)
            control_image = control_image.squeeze(0).repeat(batch_size, 1, 1, 1, 1) # [B, T, C, H, W]
            control_image = rearrange(control_image, "b t c h w -> b c t h w").contiguous()
            control_image = torch.cat([control_image] * 2) if do_classifier_free_guidance else control_image

        # Prepare latent variables
        num_channels_latents = 4
        latents_output = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
            return_image_latents=return_image_latents
        )
        if return_image_latents:
            latents, image_latents = latents_output
        else:
            latents = latents_output
        
        latents_dtype = latents.dtype

        # Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode="default", crops_coords=None
        )

        mask_condition = mask_condition.squeeze(0).repeat(batch_size, 1, 1, 1, 1) # [B, T, C, H, W]
        mask_condition = rearrange(mask_condition, "b t c h w -> b c t h w")

        
        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            init_image,
            batch_size * num_videos_per_prompt,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            do_classifier_free_guidance,
            inference_frame_number=inference_frame_number,
            deliate_mask=deliate_mask
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        context_scheduler = get_context_scheduler(context_schedule)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # context_queue = list(
                #     context_scheduler(
                #         0,
                #         num_inference_steps,
                #         latents.shape[2],
                #         context_frames,
                #         context_stride,
                #         0,
                #     )
                # )
                # num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        context_overlap,
                    )
                )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                global_context = []

                for i in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            i * context_batch_size : (i + 1) * context_batch_size
                        ]
                    )

                for context in global_context:
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    latent_mask = torch.cat([mask[:, :, c] for c in context]).to(device)
                    latent_masked_image = torch.cat([masked_image_latents[:, :, c] for c in context]).to(device)

                    if control_image is not None:
                        tmp_controlnet_cond = torch.cat([control_image[:, :, c] for c in context]).to(device)
                    else:
                        tmp_controlnet_cond = None

                    down_block_additional_residuals = mid_block_additional_residual = None
                    if (getattr(self, "controlnet", None) != None) and (tmp_controlnet_cond != None):
                        assert tmp_controlnet_cond.dim() == 5
                        
                        tmp_controlnet_cond = rearrange(tmp_controlnet_cond, "b c t h w -> (b t) c h w").contiguous()
                        controlnet_noisy_latents = rearrange(latent_model_input, "b c t h w -> (b t) c h w").contiguous()
                        controlnet_prompt_embeds = text_embeddings.repeat(context_frames, 1, 1)

                        tmp_controlnet_cond = tmp_controlnet_cond.to(latents.device)

                        # controlnet_cond_shape[2] = context_frames
                        # controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)

                        # controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
                        # controlnet_conditioning_mask_shape[1] = 1
                        # controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)

                        # assert tmp_controlnet_cond.shape[2] >= len(controlnet_image_index)
                        # controlnet_cond[:,:,controlnet_image_index] = tmp_controlnet_cond[:,:,:len(controlnet_image_index)]
                        # controlnet_conditioning_mask[:,:,controlnet_image_index] = 1


                        down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                            controlnet_noisy_latents, t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=tmp_controlnet_cond,
                            # conditioning_mask=controlnet_conditioning_mask,
                            conditioning_scale=controlnet_conditioning_scale,
                            guess_mode=False, return_dict=False,
                        )

                        down_block_additional_residuals = [rearrange(tmp, "(b t) c h w -> b c t h w", t=context_frames).contiguous() for tmp in down_block_additional_residuals]
                        mid_block_additional_residual = rearrange(mid_block_additional_residual, "(b t) c h w -> b c t h w", t=context_frames).contiguous()
                    
                    latent_model_input = torch.cat([latent_model_input, latent_mask, latent_masked_image], dim=1)
                    # predict the noise residual
                    pred = self.unet(
                        latent_model_input, t, 
                        encoder_hidden_states=text_embeddings.repeat(len(context), 1, 1),
                        down_block_additional_residuals = down_block_additional_residuals,
                        mid_block_additional_residual   = mid_block_additional_residual,
                    ).sample.to(dtype=latents_dtype)

                    if do_classifier_free_guidance:
                        pred_uc, pred_c = pred.chunk(2)
                        pred = torch.cat([pred_uc.unsqueeze(0), pred_c.unsqueeze(0)])
                        for j, c in enumerate(context):
                            noise_pred[:, :, c] = noise_pred[:, :, c] + pred[:, j]
                            counter[:, :, c] = counter[:, :, c] + 1
                    else:
                        pred = pred.unsqueeze(0)
                        for j, c in enumerate(context):
                            noise_pred[:, :, c] = noise_pred[:, :, c] + pred[:, j]
                            counter[:, :, c] = counter[:, :, c] + 1

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred / counter

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # if return_image_latents:
                #     init_latents_proper = image_latents
                #     if self.do_classifier_free_guidance:
                #         init_mask, _ = mask.chunk(2)
                #     else:
                #         init_mask = mask

                #     if i < len(timesteps) - 1:
                #         noise_timestep = timesteps[i + 1]
                #         init_latents_proper = self.scheduler.add_noise(
                #             init_latents_proper, noise, torch.tensor([noise_timestep])
                #         )

                #     latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #     progress_bar.update()
        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)

        # Post-processing
        video = self.decode_latents(latents, decoder_consistency)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
