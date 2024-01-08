"""ControlNet haven't been implemented in the given version of the diffusers. """
# from diffusers.pipelines.controlnet import MultiControlNetModel
from typing import Any, List, Union, Optional

import torch
from diffusers import UNet2DConditionModel
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available

from .attentions import CrossAttentionIPAdapter


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IPAdapterWrapper:
    def __init__(self, unet: UNet2DConditionModel, **kwargs):
        try:
            args = kwargs["args"]
        except KeyError:
            raise KeyError("args is not fed into the init function!!")
        self.unet = unet
        self.device = args.device
        self.image_encoder_path = args.image_encoder_path
        self.ip_ckpt = args.ip_ckpt
        self.num_tokens = args.num_tokens
        self.ctrl_scale = args.ctrl_scale

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()
        self.load_img_proj()
    
    def __getattr__(self, item) -> Any:
        return self.unet.__getattribute__(item)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def load_ip_adapter(self):
        unet_sd = self.unet.state_dict()
        pl_sd = torch.load(self.ip_ckpt, map_location="cpu")
        ipadapter_sd = pl_sd["ip_adapter"]
        attn_layers = {}
        i = 0
        for name, block in self.attn2_layers.items():
            if name.endswith('attn2'):
                query_dim = block.to_q.in_features
                context_dim = block.to_k.in_features 
                inner_dim = block.to_q.out_features
                heads = block.heads
                dim_head = 64 # TODO: it shall be block.dim_head
                dropout = 0. # TODO: it shall be block.dropout
                
                weights = {
                    "to_q.weight": unet_sd[name + ".to_q.weight"],
                    "to_k.weight": unet_sd[name + ".to_k.weight"],
                    "to_v.weight": unet_sd[name + ".to_v.weight"],
                    "to_out.0.weight": unet_sd[name + ".to_out.0.weight"],
                    "to_out.0.bias": unet_sd[name + ".to_out.0.bias"],
                    "to_k_ip.weight": ipadapter_sd[f"{i*2+1}.to_k_ip.weight"],
                    "to_v_ip.weight": ipadapter_sd[f"{i*2+1}.to_v_ip.weight"],
                }
                i += 1
                attn_layers[name] = CrossAttentionIPAdapter(query_dim, inner_dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout, ctrl_scale=self.ctrl_scale)
                attn_layers[name].load_state_dict(weights)
        self.set_attn2_layers(attn_layers) 
    

    def load_img_proj(self):
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        
    
    def set_attn2_layers(self, layers: dict):
        r"""
        Sets up the 2nd attention layers for each transformer block.
        """
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, layers: dict):
            if name.endswith("transformer_blocks"):
                for sub_name, child in module.named_children(): # transformer_blocks will be a ModuleList 
                    child.attn2 = layers[f"{name}.{sub_name}.attn2"]  

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, layers)

        for name, module in self.unet.named_children():
            fn_recursive_attn_processor(name, module, layers)


    @torch.inference_mode()
    def get_ctrl_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds # the size of clip_image_embeds is (bs, 1024)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds) # the size of image_prompt_embeds is (bs, 4, 1024)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds)) 
        return image_prompt_embeds, uncond_image_prompt_embeds
    

    def __call__(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,) -> Any:
        return self.unet(sample, timestep, encoder_hidden_states, class_labels, attention_mask, return_dict)  

    @property
    def attn2_layers(self) -> dict:
        r"""
        Extract the 2nd attention layer from each transformer block.
        """
        # set recursively
        layers = {}

        # def fn_recursive_add_processors(name: str, module: th.nn.Module, processors: Dict[str, AttentionProcessor]):
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, layers: dict):
            if name.endswith("attn2"):
                layers[f"{name}"] = module
            
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, layers)

            return layers

        for name, module in self.unet.named_children():
            fn_recursive_add_processors(name, module, layers)

        return layers


class ControlNetWrapper:
    pass  # TODO
