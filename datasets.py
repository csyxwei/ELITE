from packaging import version
from PIL import Image
from torchvision import transforms
import os
import PIL
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
import random
import albumentations as A
import copy
import cv2
import pandas as pd


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

def is_image(file):
    return 'jpg' in file.lower()  or 'png' in file.lower()  or 'jpeg' in file.lower()

class CustomDatasetWithBG(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        placeholder_token="*",
        template="a photo of a {}",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token

        self.image_paths = []
        self.image_paths += [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if is_image(file_path) and not 'bg' in file_path]

        self.image_paths = sorted(self.image_paths)

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.template = template

    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def process(self, image):
        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, i):
        example = {}

        placeholder_string = self.placeholder_token
        text = self.template.format(placeholder_string)
        example["text"] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        example["index"] = torch.tensor(placeholder_index)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        image = Image.open(self.image_paths[i % self.num_images])

        mask_path = self.image_paths[i % self.num_images].replace('.jpeg', '.png').replace('.jpg', '.png').replace('.JPEG', '.png')[:-4] + '_bg.png'
        mask = np.array(Image.open(mask_path))

        mask = np.where(mask > 0, 1, 0)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)
        object_tensor = image_np * mask
        example["pixel_values"] = self.process(image_np)


        ref_object_tensor = Image.fromarray(object_tensor.astype('uint8')).resize((224, 224), resample=self.interpolation)
        ref_image_tenser = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=self.interpolation)
        example["pixel_values_obj"] = self.get_tensor_clip()(ref_object_tensor)
        example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tenser)

        ref_seg_tensor = Image.fromarray(mask.astype('uint8') * 255)
        ref_seg_tensor = self.get_tensor_clip(normalize=False)(ref_seg_tensor)
        example["pixel_values_seg"] = torch.nn.functional.interpolate(ref_seg_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)

        return example


class OpenImagesDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        set="train",
        placeholder_token="*",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.set_type = set

        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])

        self.bbox_path_list = []
        if set == "train":
            bboxs_path = os.path.join(data_root, 'annotations', f'oidv6-train-annotations-bbox.csv')
        elif set == "validation":
            bboxs_path = os.path.join(data_root, 'annotations', f'validation-annotations-bbox.csv')
        else:
            bboxs_path = os.path.join(data_root, 'annotations', f'test-annotations-bbox.csv')

        df_val_bbox = pd.read_csv(bboxs_path)
        bbox_groups = df_val_bbox.groupby(df_val_bbox.LabelName)

        bbox_full = []
        for label_name in df_val_bbox['LabelName'].unique():
            bboxs = bbox_groups.get_group(label_name)[
                ['XMin', 'XMax', 'YMin', 'YMax', 'LabelName', 'ImageID',
                 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsInside']].values.tolist()
            bboxs_new = []
            for bbox in bboxs:
                if not ((bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) > 0.8 or (bbox[1] - bbox[0]) * (
                        bbox[3] - bbox[2]) < 0.02):
                    bboxs_new.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]])
            bbox_full.extend(bboxs_new)

        self.bboxs_full = bbox_full

        self.num_images = len(bbox_full)

        print('{}: total {} images ...'.format(set, self.num_images))

        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_templates_small


    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def process(self, image):
        img = np.array(image)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def obtain_text(self, add_caption, object_category=None):

        if object_category is None:
            placeholder_string = self.placeholder_token
        else:
            placeholder_string = object_category

        text = random.choice(self.templates).format(placeholder_string)
        text = add_caption + text[1:]

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        index = torch.tensor(placeholder_index)

        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return input_ids, index, text

    def __getitem__(self, i):
        example = {}

        input_ids, index, text = self.obtain_text('a')
        example["input_ids"] = input_ids
        example["index"] = index
        example["text"] = text

        bbox_sample = self.bboxs_full[i % self.num_images]
        bbox_sample = copy.copy(bbox_sample)

        file_name = bbox_sample[-1] + '.jpg'
        img_path = os.path.join(self.data_root, 'images', self.set_type, file_name)

        try:
            img_p = Image.open(img_path).convert("RGB")
            img_p_np = np.array(img_p)
            bbox_sample[0] *= int(img_p_np.shape[1])
            bbox_sample[1] *= int(img_p_np.shape[1])
            bbox_sample[2] *= int(img_p_np.shape[0])
            bbox_sample[3] *= int(img_p_np.shape[0])

            bbox_pad = copy.copy(bbox_sample)
            bbox_pad[0] = int(bbox_sample[0] - min(10, bbox_sample[0] - 0))
            bbox_pad[1] = int(bbox_sample[1] + min(10, img_p.size[0] - bbox_sample[1]))
            bbox_pad[2] = int(bbox_sample[2] - min(10, bbox_sample[2] - 0))
            bbox_pad[3] = int(bbox_sample[3] + min(10, img_p.size[1] - bbox_sample[3]))

            image_tensor = img_p_np[bbox_pad[2]:bbox_pad[3], bbox_pad[0]:bbox_pad[1], :]
            example["pixel_values"] = self.process(image_tensor)

            ref_image_tensor = self.random_trans(image=image_tensor)
            ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
            example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tensor)

        except Exception as e:
            example["pixel_values"] = torch.zeros((3, 512, 512))
            example["pixel_values_clip"] = torch.zeros((3, 224, 224))
            with open('error.txt', 'a+') as f:
                f.write(str(e) + '\n')

        return example


class OpenImagesDatasetWithMask(OpenImagesDataset):
    def __init__(self,
             data_root,
             tokenizer,
             size=512,
             interpolation="bicubic",
             set="train",
             placeholder_token="*"):

        # super().__init__(data_root, tokenizer, size, interpolation, set, placeholder_token)
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.set = set

        class_anno_path = os.path.join(data_root, 'annotations', f'oidv6-class-descriptions.csv')
        anno_files = pd.read_csv(class_anno_path)
        class_groups = anno_files.groupby(anno_files.LabelName)

        if set == "train":
            bboxs_path = os.path.join(data_root, 'annotations', f'train-annotations-object-segmentation.csv')
            dict_path = os.path.join(data_root, 'segs', f'train_bbox_dict.npy')
        elif set == "validation":
            bboxs_path = os.path.join(data_root, 'annotations', f'validation-annotations-object-segmentation.csv')
            dict_path = os.path.join(data_root, 'segs', f'validation_bbox_dict.npy')
        else:
            bboxs_path = os.path.join(data_root, 'annotations', f'test-annotations-object-segmentation.csv')
            dict_path = os.path.join(data_root, 'segs', f'test_bbox_dict.npy')

        bbox_dict = np.load(dict_path, allow_pickle=True).item()

        df_val_bbox = pd.read_csv(bboxs_path)
        bbox_groups = df_val_bbox.groupby(df_val_bbox.LabelName)
        bboxes_full = []
        for label_name in df_val_bbox['LabelName'].unique():
            bboxs = bbox_groups.get_group(label_name)[
                ['BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax', 'LabelName', 'MaskPath']].values.tolist()
            bboxes_new = []
            for box in bboxs:
                if not box[-1] in bbox_dict:
                    continue
                bbox_data = bbox_dict[box[-1]]

                if (bbox_data[2] - bbox_data[1]) < 100 or (bbox_data[4] - bbox_data[3]) < 100:
                    continue
                if not ((bbox_data[2] - bbox_data[1]) / (bbox_data[4] - bbox_data[3]) < 0.5 or (
                        bbox_data[4] - bbox_data[3]) / ( bbox_data[2] - bbox_data[1]) < 0.5):
                    class_name = class_groups.get_group(box[4])[['DisplayName']].values.tolist()[0][0]
                    bboxes_new.append([box[-1], bbox_data[1], bbox_data[2], bbox_data[3], bbox_data[4], class_name])

            bboxes_full.extend(bboxes_new)

        self.bboxes_full = bboxes_full
        self.num_images = len(bboxes_full)

        print('{}: total {} images ...'.format(set, self.num_images))

        self._length = self.num_images
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_templates_small


    def __len__(self):
        return self._length

    ## borrowed from custom diffusion
    def custom_aug(self, instance_image):
        instance_image = Image.fromarray(instance_image)
        #### apply augmentation and create a valid image regions mask ####
        if np.random.randint(0, 3) < 2:
            random_scale = np.random.randint(self.size // 3, self.size + 1)
        else:
            random_scale = np.random.randint(int(1.2 * self.size), int(1.4 * self.size))

        if random_scale % 2 == 1:
            random_scale += 1

        if random_scale < 0.6 * self.size:
            add_to_caption = np.random.choice(["a far away", "very small"])
            cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
            cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

            instance_image1 = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image1 = np.array(instance_image1).astype(np.uint8)
            instance_image1 = (instance_image1 / 127.5 - 1.0).astype(np.float32)

            instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
            instance_image[cx - random_scale // 2: cx + random_scale // 2,
            cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1

            mask = np.zeros((self.size // 8, self.size // 8))
            mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1,
            (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.

        elif random_scale > self.size:
            add_to_caption = np.random.choice(["zoomed in", "close up"])
            cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
            cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

            instance_image = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            instance_image = instance_image[cx - self.size // 2: cx + self.size // 2,
                             cy - self.size // 2: cy + self.size // 2, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            add_to_caption = "a"
            if self.size is not None:
                instance_image = instance_image.resize((self.size, self.size), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))

        return torch.from_numpy(instance_image).permute(2, 0, 1), torch.from_numpy(mask[:, :, None]).permute(2, 0, 1), add_to_caption

    def aug_cv2(self, img, seg):

        img_auged = np.array(img).copy()
        seg_auged = np.array(seg).copy()
        # resize and crop
        if random.choice([0, 1]) == 0:
            new_size = random.randint(224, 256)
            img_auged = cv2.resize(img_auged, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
            seg_auged = cv2.resize(seg_auged, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

            start_x, start_y = random.randint(0, new_size - 224), random.randint(0, new_size - 224)
            img_auged = img_auged[start_x:start_x + 224, start_y:start_y + 224, :]
            seg_auged = seg_auged[start_x:start_x + 224, start_y:start_y + 224, :]

        h, w = img_auged.shape[:2]
        # rotate
        if random.choice([0, 1]) == 0:
            # print('rotate')
            angle = random.randint(-30, 30)
            M = cv2.getRotationMatrix2D((112, 112), angle, 1)
            img_auged = cv2.warpAffine(img_auged, M, (w, h), flags=cv2.INTER_CUBIC)
            seg_auged = cv2.warpAffine(seg_auged, M, (w, h), flags=cv2.INTER_NEAREST)

        # translation
        if random.choice([0, 1]) == 0:
            trans_x = random.randint(-60, 60)
            trans_y = random.randint(-60, 60)
            H = np.float32([[1, 0, trans_x],
                            [0, 1, trans_y]])
            img_auged = cv2.warpAffine(img_auged, H, (w, h), flags=cv2.INTER_CUBIC)
            seg_auged = cv2.warpAffine(seg_auged, H, (w, h), flags=cv2.INTER_NEAREST)

        img_auged = Image.fromarray(img_auged)
        seg_auged = Image.fromarray(seg_auged)

        return img_auged, seg_auged


    def __getitem__(self, i):
        example = {}

        seg_name = self.bboxes_full[i % self.num_images][0]
        file_name = seg_name.split('_')[0] + '.jpg'
        img_path = os.path.join(self.data_root, 'images', self.set, file_name)
        seg_path = os.path.join(self.data_root, 'segs', self.set, seg_name)

        try:
            # crop image and mask
            bbox_sample = self.bboxes_full[i % self.num_images][1:]
            img_p_np = cv2.imread(img_path)
            img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
            seg_p_np = cv2.imread(seg_path).astype('float')
            seg_p_np = cv2.resize(seg_p_np, img_p_np.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

            bbox_pad = copy.copy(bbox_sample)
            pad_size = random.choice(list(range(10, 20)))
            bbox_pad[0] = int(bbox_pad[0] - min(pad_size, bbox_pad[0] - 0))
            bbox_pad[1] = int(bbox_pad[1] + pad_size)
            bbox_pad[2] = int(bbox_pad[2] - min(pad_size, bbox_pad[2] - 0))
            bbox_pad[3] = int(bbox_pad[3] + pad_size)

            image_tensor = img_p_np[bbox_pad[0]:bbox_pad[1], bbox_pad[2]:bbox_pad[3], :]
            seg_tensor = seg_p_np[bbox_pad[0]:bbox_pad[1], bbox_pad[2]:bbox_pad[3], :]

            # augmentation for input image
            augged_image, augged_mask, add_caption = self.custom_aug(image_tensor)
            input_ids, index, text = self.obtain_text(add_caption)

            example["pixel_values"] = augged_image
            example["mask_values"] = augged_mask
            example["input_ids"] = input_ids
            example["index"] = index
            example["text"] = text

            object_tensor = image_tensor * (seg_tensor / 255)
            ref_object_tensor = cv2.resize(object_tensor, (224, 224), interpolation=cv2.INTER_CUBIC)
            ref_image_tenser = cv2.resize(image_tensor, (224, 224), interpolation=cv2.INTER_CUBIC)
            ref_seg_tensor = cv2.resize(seg_tensor, (224, 224), interpolation=cv2.INTER_NEAREST)

            ref_object_tensor, ref_seg_tensor = self.aug_cv2(ref_object_tensor.astype('uint8'), ref_seg_tensor.astype('uint8'))
            example["pixel_values_clip"] = self.get_tensor_clip()(Image.fromarray(ref_image_tenser))
            example["pixel_values_obj"] = self.get_tensor_clip()(ref_object_tensor)
            example["pixel_values_seg"] = self.get_tensor_clip(normalize=False)(ref_seg_tensor)

        except Exception as e:
            example["pixel_values"] = torch.zeros((3, 512, 512))
            example["pixel_values_obj"] = torch.zeros((3, 224, 224))
            example["pixel_values_clip"] = torch.zeros((3, 224, 224))
            example["pixel_values_seg"] = torch.zeros((3, 224, 224))

            input_ids, index, text = self.obtain_text("a")
            example["input_ids"] = input_ids
            example["index"] = index
            example["text"] = text

            with open('error.txt', 'a+') as f:
                f.write(str(e) + '\n')

        return example
