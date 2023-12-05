"""For caching SD v1-5 for the inference scripts fail to cache the models."""
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(
    repo_id,
    use_safetensors=True,
)
