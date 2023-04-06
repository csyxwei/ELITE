from __future__ import annotations
import pathlib
import gradio as gr
import torch
import PIL
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Any

from inference_local import pww_load_tools, validation
from train_local import LMSDiscreteScheduler

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [T.ToTensor()]
    if normalize:
        transform_list += [
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ]
    return T.Compose(transform_list)


def process(image: np.ndarray, size: int = 512) -> torch.Tensor:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = np.array(image).astype(np.float32)
    image = image / 127.5 - 1.0
    return torch.from_numpy(image).permute(2, 0, 1)

class Model:
    def __init__(self,
                 pretrained_model_name_or_path: str='CompVis/stable-diffusion-v1-4',
                 global_mapper_path: str='./checkpoints/global_mapper.pt',
                 local_mapper_path: str='./checkpoints/local_mapper.pt'):

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.vae, self.unet, self.text_encoder, self.tokenizer, self.image_encoder, self.mapper, self.mapper_local, self.scheduler = pww_load_tools(
            self.device,
            LMSDiscreteScheduler,
            diffusion_model_path=pretrained_model_name_or_path,
            mapper_model_path=global_mapper_path,
            mapper_local_model_path=local_mapper_path,
        )

    def prepare_data(self,
                     image: PIL.Image.Image,
                     mask: PIL.Image.Image,
                     text: str,
                     placeholder_string: str = 'S') -> dict[str, Any]:

        data: dict[str, Any] = {}

        data['text'] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        data['index'] = torch.tensor(placeholder_index)

        data['input_ids'] = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        ).input_ids[0]

        image = image.convert('RGB')
        mask = mask.convert('RGB')
        mask = np.array(mask) / 255.0

        image_np = np.array(image)
        object_tensor = image_np * mask
        data['pixel_values'] = process(image_np)

        ref_object_tensor = PIL.Image.fromarray(
            object_tensor.astype('uint8')).resize(
            (224, 224), resample=PIL.Image.Resampling.BICUBIC)
        ref_image_tenser = PIL.Image.fromarray(
            image_np.astype('uint8')).resize(
            (224, 224), resample=PIL.Image.Resampling.BICUBIC)
        data['pixel_values_obj'] = get_tensor_clip()(ref_object_tensor)
        data['pixel_values_clip'] = get_tensor_clip()(ref_image_tenser)

        ref_seg_tensor = PIL.Image.fromarray(mask.astype('uint8') * 255)
        ref_seg_tensor = get_tensor_clip(normalize=False)(ref_seg_tensor)
        data['pixel_values_seg'] = F.interpolate(ref_seg_tensor.unsqueeze(0),
                                                 size=(128, 128),
                                                 mode='nearest').squeeze(0)
        device = torch.device(self.device)
        data['pixel_values'] = data['pixel_values'].to(device)
        data['pixel_values_clip'] = data['pixel_values_clip'].to(device).half()
        data['pixel_values_obj'] = data['pixel_values_obj'].to(device).half()
        data['pixel_values_seg'] = data['pixel_values_seg'].to(device).half()
        data['input_ids'] = data['input_ids'].to(device)
        data['index'] = data['index'].to(device).long()

        for key, value in list(data.items()):
            if isinstance(value, torch.Tensor):
                data[key] = value.unsqueeze(0)

        return data

    def run(self,
            image: dict[str, PIL.Image.Image],
            text: str,
            seed: int,
            guidance_scale: float,
            lambda_: float,
            num_steps: int,):

        example = self.prepare_data(image['image'], image['mask'], text)

        if seed == -1:
            seed = np.random.randint(0, 1000000)

        image = validation(example, self.tokenizer, self.image_encoder, self.text_encoder, self.unet, self.mapper, self.mapper_local, self.vae,
                            example["pixel_values_clip"].device, guidance_scale,
                            seed=seed, llambda=float(lambda_), num_steps=num_steps)
        return image[0]

def create_demo():
    TITLE = '# [ELITE Demo](https://github.com/csyxwei/ELITE)'

    USAGE = '''To run the demo, you should:   
    1. Upload your image.   
    2. **Draw a mask on the object part.**   
    3. Input proper text prompts, such as "A photo of S" or "A S wearing sunglasses", where "S" denotes your customized concept.   
    4. Click the Run button. You can also adjust the hyperparameters to improve the results.
    '''

    model = Model()

    with gr.Blocks() as demo:
        gr.Markdown(TITLE)
        gr.Markdown(USAGE)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    image = gr.Image(label='Input', tool='sketch', type='pil')
                    # gr.Markdown('Draw a mask on your object.')
                    gr.Markdown(
                        'Upload your image and **draw a mask on the object part.** Like [this](https://user-images.githubusercontent.com/23421814/224873479-c4cf44d6-8c99-4ef9-b972-87c25fe923ee.png).')
                prompt = gr.Text(
                    label='Prompt',
                    placeholder='e.g. "A photo of S", "A S wearing sunglasses"',
                    info='Use "S" for your concept.')
                lambda_ = gr.Slider(
                    label='Lambda',
                    minimum=0,
                    maximum=1.5,
                    step=0.1,
                    value=0.6,
                    info=
                    'The larger the lambda, the more consistency between the generated image and the input image, but less editability.'
                )
                run_button = gr.Button('Run')
                with gr.Accordion(label='Advanced options', open=False):
                    seed = gr.Slider(
                        label='Seed',
                        minimum=-1,
                        maximum=1000000,
                        step=1,
                        value=-1,
                        info=
                        'If set to -1, a different seed will be used each time.'
                    )
                    guidance_scale = gr.Slider(label='Guidance scale',
                                               minimum=0,
                                               maximum=50,
                                               step=0.1,
                                               value=5.0)
                    num_steps = gr.Slider(
                        label='Steps',
                        minimum=1,
                        maximum=300,
                        step=1,
                        value=100,
                        info=
                        'In the paper, the number of steps is set to 100, but in this demo the default value is 20 to reduce inference time.'
                    )
            with gr.Column():
                result = gr.Image(label='Result')

        paths = sorted([
            path.as_posix()
            for path in pathlib.Path('./test_datasets').glob('*')
            if 'bg' not in path.stem
        ])
        gr.Examples(examples=paths, inputs=image, examples_per_page=20)

        inputs = [
            image,
            prompt,
            seed,
            guidance_scale,
            lambda_,
            num_steps,
        ]
        prompt.submit(fn=model.run, inputs=inputs, outputs=result)
        run_button.click(fn=model.run, inputs=inputs, outputs=result)
    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.queue(api_open=False).launch()