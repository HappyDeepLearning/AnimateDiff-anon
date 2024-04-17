import os, io, csv, math, random
import numpy as np
from glob import glob
from einops import rearrange
import torch
from kornia import morphology as morph

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import pickle
from natsort import natsorted


def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray, is_sil=False) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor.
        images: (T, C, H, W) or (T, H, W)
    """
    images = torch.from_numpy(images)
    if is_sil:
        mask = images.float() / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        return mask
    else:
        return images.float() / 127.5 - 1.0
        # return images.float() / 255


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        """
        seq: (C, H, W)
        """
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[..., ::-1]

def erode_dilate(mask, erode_dilate_flag=None):
    """"
    mask range: 0-1
    """
    if len(mask.size()) == 3:
        flag = True
        mask = mask[None, ...]
    else:
        flag = False
    kernel_size = random.choice(range(5, 10))
    kernel = torch.ones((kernel_size, kernel_size))
    if erode_dilate_flag is None:
        erode_dilate_flag = random.choice([0,1])
    if erode_dilate_flag == 0:
        mask_erosion = morph.erosion(mask, kernel)
        mask=1*(mask_erosion>0)
    else:
        mask_dilation = morph.dilation(mask, kernel)
        mask=1*(mask_dilation>0)
    
    if flag:
        mask = mask.squeeze(0)

    return mask.float()

class GaitLUDatasetImg(Dataset):

    def __init__(self, data_root, sample_size=256, text_prompts=None, seed=347, is_train=True, train_test_ratio=0.8, **kwargs):
        random.seed(seed)

        self.all_ids_list_rgb = sorted(glob(os.path.join(data_root, f"*/org/*/*")))
        self.all_ids_list_sil = sorted(glob(os.path.join(data_root, f"*/sil/*/*")))


        self.all_list_rgb, self.all_list_sil = [], []

        assert len(self.all_ids_list_rgb) == len(self.all_ids_list_rgb)

        total_length = len(self.all_ids_list_rgb)
        data_length = int(total_length * train_test_ratio)
        data_indices = random.sample(range(total_length), data_length)
        if not is_train:
            new_data_length = total_length - data_length
            new_data_indices = list(set(range(total_length)) - set(data_indices))
            data_indices = new_data_indices
            data_length = new_data_length

        for i in range(total_length):
            if i not in data_indices:
                continue
            self.all_list_rgb += sorted(glob(os.path.join(self.all_ids_list_rgb[i], "*.png")))
            self.all_list_sil += sorted(glob(os.path.join(self.all_ids_list_sil[i], "*.png")))
        self.length = len(self.all_list_rgb)

        self.pixel_transforms = RandomHorizontalFlip(prob=0.5)
        self.resize = transforms.Resize(sample_size)
        self.text_prompts = text_prompts
        
    
    def __len__(self):
        return self.length
    
    def load_image(self, image_path, is_sil=False, resize=transforms.Resize(256)):
        img = Image.open(image_path)
        if is_sil:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = resize(img)
        img = np.array(img)

        if is_sil:
            img = img[..., None]
        return img

    def __getitem__(self, idx):
        rgb_path = self.all_list_rgb[idx]
        sil_path = self.all_list_sil[idx]

        rgb_data = self.load_image(rgb_path, resize=self.resize)
        sil_data = self.load_image(sil_path, is_sil=True, resize=self.resize)

        # transform
        all_data = np.ascontiguousarray(np.concatenate([rgb_data, sil_data], axis=-1).transpose(2, 0, 1)) # (C, H, W)
        all_data = np.ascontiguousarray(self.pixel_transforms(all_data))

        rgb_data = numpy_to_pt(all_data[:3]) # (3, H, W), (-1, 1)
        sil_data = numpy_to_pt(all_data[3:], is_sil=True) # (1, H, W), (0-1)
        
        if random.random() < 0.5 :
            new_sil_data = torch.zeros_like(sil_data)
            tmp_mean_value = rgb_data.mean(0, keepdim=True)
            new_sil_data[tmp_mean_value!=-1] = 1
            if random.random() < 0.5:
                sil_data = erode_dilate(new_sil_data, erode_dilate_flag=0)
            else:
                sil_data = new_sil_data
        else:
            if random.random() < 0.5:
                sil_data = erode_dilate(sil_data, erode_dilate_flag=1)

        # if random.random() < 0.5:
        #     sil_data = erode_dilate(sil_data, erode_dilate_flag=1)

        fg_data = rgb_data * (sil_data < 0.5) # only background

        if self.text_prompts is not None:
            prob_each_parts = self.text_prompts["prob_each_parts"]
            base_text = self.text_prompts["base_text"]
            adj_text = self.text_prompts["adj_text"]
            n_text = self.text_prompts["n_text"]
            v_text = self.text_prompts["v_text"]
            adv_text = self.text_prompts["adv_text"]
            base_text_prompt = random.choice(base_text) if random.random() < prob_each_parts[0] else ""
            adj_text_prompt = random.choice(adj_text) if random.random() < prob_each_parts[1] else ""
            n_text_prompt = random.choice(n_text) if random.random() < prob_each_parts[2] else ""
            v_text_prompt = random.choice(v_text) if random.random() < prob_each_parts[3] else ""
            adv_text_prompt = random.choice(adv_text) if random.random() < prob_each_parts[4] else ""
            text = f"{base_text_prompt} {adj_text_prompt} {n_text_prompt} {v_text_prompt} {adv_text_prompt}"
            text = text.strip()
        else:
            text = "(person, full body), best quality, highres"
        sample = dict(pixel_values=rgb_data, fg_pixel_values=fg_data, text=text, sil_pixel_values=sil_data)
        return sample

        
if __name__ == "__main__":

    config_path = "configs/training/anon/stage1_image_finetune_inpaint.yaml"

    from omegaconf import OmegaConf
    args = config = OmegaConf.load(config_path)
    train_data_kwargs = args.train_data

    dataset = GaitLUDatasetImg(
        **train_data_kwargs,
        is_train=True,
        train_test_ratio=0.8
    )

    # train_ratio = 0.8  # 训练集占总数据集的比例
    # test_ratio = 1 - train_ratio  # 测试集占总数据集的比例

    # train_size = int(train_ratio * len(dataset))
    # test_size = len(dataset) - train_size

    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(len(dataset))
    train_dataloader = DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(train_dataloader):
        print(f"========================{idx:06d}========================")
        print(batch["pixel_values"].shape)
        print(batch["sil_pixel_values"].shape)
        print(batch["fg_pixel_values"].shape)
        print(batch["text"])
        if idx == 10:
            break
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)