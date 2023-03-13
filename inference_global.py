import os
from typing import Optional, Tuple
import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from train_global import Mapper, th2image
from train_global import inj_forward_text, inj_forward_crossattention, validation
import torch.nn as nn
from datasets import CustomDatasetWithBG

def _pil_from_latents(vae, latents):
    _latents = 1 / 0.18215 * latents.clone()
    image = vae.decode(_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    ret_pil_images = [Image.fromarray(image) for image in images]

    return ret_pil_images


def pww_load_tools(
    device: str = "cuda:0",
    scheduler_type=LMSDiscreteScheduler,
    mapper_model_path: Optional[str] = None,
    diffusion_model_path: Optional[str] = None,
    model_token: Optional[str] = None,
) -> Tuple[
    UNet2DConditionModel,
    CLIPTextModel,
    CLIPTokenizer,
    AutoencoderKL,
    CLIPVisionModel,
    Mapper,
    LMSDiscreteScheduler,
]:

    # 'CompVis/stable-diffusion-v1-4'
    local_path_only = diffusion_model_path is not None
    vae = AutoencoderKL.from_pretrained(
        diffusion_model_path,
        subfolder="vae",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only,
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)


    # Load models and create wrapper for stable diffusion
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    unet = UNet2DConditionModel.from_pretrained(
        diffusion_model_path,
        subfolder="unet",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only,
    )

    mapper = Mapper(input_dim=1024, output_dim=768)

    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            if 'attn1' in _name: continue
            _module.__class__.__call__ = inj_forward_crossattention

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

    mapper.load_state_dict(torch.load(mapper_model_path, map_location='cpu'))
    mapper.half()

    for _name, _module in unet.named_modules():
        if 'attn1' in _name: continue
        if _module.__class__.__name__ == "CrossAttention":
            _module.add_module('to_k_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_k'))
            _module.add_module('to_v_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_v'))

    vae.to(device), unet.to(device), text_encoder.to(device), image_encoder.to(device), mapper.to(device)

    scheduler = scheduler_type(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    vae.eval()
    unet.eval()
    image_encoder.eval()
    text_encoder.eval()
    mapper.eval()
    return vae, unet, text_encoder, tokenizer, image_encoder, mapper, scheduler


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--token_index",
        type=str,
        default="full",
        help="Selected index for word embedding.",
    )

    parser.add_argument(
        "--global_mapper_path",
        type=str,
        required=True,
        help="Path to pretrained global mapping network.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='outputs',
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="S",
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--template",
        type=str,
        default="a photo of a {}",
        help="Text template for customized genetation.",
    )

    parser.add_argument(
        "--test_data_dir", type=str, default=None, required=True, help="A folder containing the testing data."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="object",
        help="Suffix of save directory.",
    )

    parser.add_argument(
        "--selected_data",
        type=int,
        default=-1,
        help="Data index. -1 for all.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for testing.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    save_dir = os.path.join(args.output_dir, f'{args.suffix}_token{args.token_index}')
    os.makedirs(save_dir, exist_ok=True)

    vae, unet, text_encoder, tokenizer, image_encoder, mapper, scheduler = pww_load_tools(
            "cuda:0",
            LMSDiscreteScheduler,
            diffusion_model_path=args.pretrained_model_name_or_path,
            mapper_model_path=args.global_mapper_path,
        )

    train_dataset = CustomDatasetWithBG(
        data_root=args.test_data_dir,
        tokenizer=tokenizer,
        size=512,
        placeholder_token=args.placeholder_token,
        template=args.template,
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    for step, batch in enumerate(train_dataloader):
        if args.selected_data > -1 and step != args.selected_data:
            continue
        batch["pixel_values"] = batch["pixel_values"].to("cuda:0")
        batch["pixel_values_clip"] = batch["pixel_values_clip"].to("cuda:0").half()
        batch["input_ids"] = batch["input_ids"].to("cuda:0")
        batch["index"] = batch["index"].to("cuda:0").long()
        print(step, batch['text'])
        syn_images = validation(batch, tokenizer, image_encoder, text_encoder, unet, mapper, vae, batch["pixel_values_clip"].device, 5,
                                token_index=args.token_index, seed=args.seed)
        concat = np.concatenate((np.array(syn_images[0]), th2image(batch["pixel_values"][0])), axis=1)
        Image.fromarray(concat).save(os.path.join(save_dir, f'{str(step).zfill(5)}_{str(args.seed).zfill(5)}.jpg'))