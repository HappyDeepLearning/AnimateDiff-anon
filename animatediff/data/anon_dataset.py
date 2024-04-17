import os, io, csv, math, random
import numpy as np
from glob import glob
from einops import rearrange
import torch
from kornia import morphology as morph

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
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
        seq: (T, C, H, W)
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

class GaitLUDatasetPKL(Dataset):

    def __init__(self, rgb_root, pose_root, sil_root, frame_number=8, interval=4, sample_size=256, image_finetune=False, text_prompts=None):
        city_name="*"
        self.all_ids_list_rgb = sorted(glob(os.path.join(rgb_root, f"{city_name}/*.pkl")))
        self.all_ids_list_pose = sorted(glob(os.path.join(pose_root, f"{city_name}/*.pkl")))
        self.all_ids_list_sil = sorted(glob(os.path.join(sil_root, f"{city_name}/*.pkl")))

        assert len(self.all_ids_list_rgb) == len(self.all_ids_list_pose)
        assert len(self.all_ids_list_rgb) == len(self.all_ids_list_sil)

        self.seq_length = frame_number
        self.interval = interval
        self.pixel_transforms = RandomHorizontalFlip(prob=0.5)
        self.resize = transforms.Resize(sample_size)
        self.length = len(self.all_ids_list_rgb)
        self.image_finetune = image_finetune
        self.text_prompts = text_prompts
    
    def __len__(self):
        return self.length

    def load_pkl(self, path):            
        with open(path, "rb") as f:
            data = pickle.load(f)

        data = np.array(data)
        return data
    
    def preprocess(self, data, is_sil=False, resize=transforms.Resize(256), indices=None):
        if is_sil:
            img_mode = "L"
        else:
            img_mode = "RGB"
        data, indices = self.same_frames(data, indices=indices)
        resized_data_list = [np.array(resize(Image.fromarray(tmp).convert(img_mode))) for tmp in data]
        resized_data = np.array(resized_data_list)
        if is_sil:
            # if self.image_finetune:
            resized_data = resized_data[..., None]
        return resized_data, indices
        


        # path, indices = self.same_frames(path, indices=indices)
        # images = []
        # for img_path in path:
        #     img = Image.open(img_path)
        #     if is_sil:
        #         img = img.convert("L")
        #     else:
        #         img = img.convert("RGB")
        #     img = resize(img)
        #     img = np.array(img)
        #     images.append(img)
        # if is_sil:
        #     images = np.array(images)[..., None]
        # else:
        #     images = np.array(images)
        # return images, indices

    def same_frames(self, seqs, indices=None):
        if indices is not None:
            if self.image_finetune:
                return seqs[indices][np.newaxis, ...], indices
            else:
                return seqs[indices], indices
        if self.image_finetune:
            indices = random.sample(list(range(seqs.shape[0])), k=1)[0]
            return seqs[indices][np.newaxis, ...], indices
        seq_len = seqs.shape[0]
        indices = list(range(seq_len))
        fs_n = self.seq_length + self.interval
        if seq_len < fs_n:
            it = math.ceil(fs_n / seq_len)
            seq_len = seq_len * it
            indices = indices * it

        start = random.choice(list(range(0, seq_len - fs_n + 1)))
        end = start + fs_n
        idx_lst = list(range(seq_len))
        idx_lst = idx_lst[start:end]
        idx_lst = sorted(np.random.choice(
            idx_lst, self.seq_length, replace=False))
        indices = [indices[i] for i in idx_lst]
        return seqs[indices], indices


    def __getitem__(self, idx):
        rgb_path = self.all_ids_list_rgb[idx]
        pose_path = self.all_ids_list_pose[idx]
        sil_path = self.all_ids_list_sil[idx]

        
        rgb_data = self.load_pkl(rgb_path)
        pose_data = self.load_pkl(pose_path)
        sil_data = self.load_pkl(sil_path)

        length_rgb = rgb_data.shape[0]
        length_pose = pose_data.shape[0]
        length_sil = sil_data.shape[0]

        min_length = min(length_rgb, length_pose, length_sil)

        if min_length == length_rgb:
            rgb_data, indices = self.preprocess(rgb_data, resize=self.resize, indices=None)
            pose_data, _ = self.preprocess(pose_data, resize=self.resize, indices=indices)
            sil_data, _ = self.preprocess(sil_data, is_sil=True, resize=self.resize, indices=indices)
        elif min_length == length_pose:
            pose_data, indices = self.preprocess(pose_data, resize=self.resize, indices=None)
            rgb_data, _ = self.preprocess(rgb_data, resize=self.resize, indices=indices)
            sil_data, _ = self.preprocess(sil_data, is_sil=True, resize=self.resize, indices=indices)
        else:
            sil_data, indices = self.preprocess(sil_data, is_sil=True, resize=self.resize, indices=None)
            rgb_data, _ = self.preprocess(rgb_data, resize=self.resize, indices=indices)
            pose_data, _ = self.preprocess(pose_data, resize=self.resize, indices=indices)
        

        # transform
        all_data = np.ascontiguousarray(np.concatenate([rgb_data, pose_data, sil_data], axis=-1).transpose(0, 3, 1, 2)) # (T, C, H, W)
        all_data = np.ascontiguousarray(self.pixel_transforms(all_data))

        if self.image_finetune:
            rgb_data = numpy_to_pt(all_data[0, :3]) # (T, 3, H, W), (-1, 1)
            pose_data = numpy_to_pt(all_data[0, 3:6]) # (T, 3, H, W), (-1, 1)
            sil_data = numpy_to_pt(all_data[0, 6:], is_sil=True) # (T, 1, H, W), (0-1)
        else:
            rgb_data = numpy_to_pt(all_data[:, :3]) # (T, 3, H, W), (-1, 1)
            pose_data = numpy_to_pt(all_data[:, 3:6]) # (T, 3, H, W), (-1, 1)
            sil_data = numpy_to_pt(all_data[:, 6:], is_sil=True) # (T, 1, H, W), (0-1)
        
        if random.random() < 0.5 :
            new_sil_data = torch.zeros_like(sil_data)
            if len(rgb_data.size()) == 3:
                tmp_mean_value = rgb_data.mean(0, keepdim=True)
            else:
                tmp_mean_value = rgb_data.mean(1, keepdim=True)
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
            text = random.choice(self.text_prompts)
            prompts_list = ["best quality", "photo-realistic", "detailed clothes"]
            text = f"{text}, {random.choice(prompts_list)}"
        else:
            text = "(person, full body), best quality, highres"
        sample = dict(pixel_values=rgb_data, pose_pixel_values=pose_data, fg_pixel_values=fg_data, text=text, sil_pixel_values=sil_data)
        return sample


class GaitLUDatasetImg(Dataset):

    def __init__(self, rgb_root, pose_root, sil_root, frame_number=8, interval=4, sample_size=256, image_finetune=False, text_prompts=None):
        city_name = '*'
        self.all_ids_list_rgb = sorted(glob(os.path.join(rgb_root, f"{city_name}/*")))
        self.all_ids_list_pose = sorted(glob(os.path.join(pose_root, f"{city_name}/*")))
        self.all_ids_list_sil = sorted(glob(os.path.join(sil_root, f"{city_name}/*")))

        assert len(self.all_ids_list_rgb) == len(self.all_ids_list_pose)
        assert len(self.all_ids_list_rgb) == len(self.all_ids_list_sil)

        self.seq_length = frame_number
        self.interval = interval
        self.pixel_transforms = RandomHorizontalFlip(prob=0.5)
        self.resize = transforms.Resize(sample_size)
        self.length = len(self.all_ids_list_rgb)
        self.image_finetune = image_finetune
        self.text_prompts = text_prompts
    
    def __len__(self):
        return self.length
    
    def load_images(self, path, is_sil=False, resize=transforms.Resize(256), indices=None):
        path, indices = self.same_frames(path, indices=indices)
        images = []
        for img_path in path:
            img = Image.open(img_path)
            if is_sil:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            img = resize(img)
            img = np.array(img)
            images.append(img)
        if is_sil:
            images = np.array(images)
            images = images[..., None]
        else:
            images = np.array(images)
        return images, indices

    # def same_frames(self, seqs, indices=None):
    #     if indices is not None:
    #         return seqs[indices]
    #     seq_len = seqs.shape[0]
    #     indices = list(range(seq_len))
    #     fs_n = self.seq_length + self.interval
    #     if seq_len < fs_n:
    #         it = math.ceil(fs_n / seq_len)
    #         seq_len = seq_len * it
    #         indices = indices * it

    #     start = random.choice(list(range(0, seq_len - fs_n + 1)))
    #     end = start + fs_n
    #     idx_lst = list(range(seq_len))
    #     idx_lst = idx_lst[start:end]
    #     idx_lst = sorted(np.random.choice(
    #         idx_lst, self.seq_length, replace=False))
    #     indices = [indices[i] for i in idx_lst]
    #     return seqs[indices], indices

    def same_frames(self, images_path, indices=None):
        if indices is not None:
            return [images_path[i] for i in indices], indices
        if self.image_finetune:
            indices = random.sample(list(range(len(images_path))), k=1)
            return [images_path[i] for i in indices], indices
        seq_len = len(images_path)
        indices = list(range(seq_len))
        fs_n = self.seq_length + self.interval
        if seq_len < fs_n:
            it = math.ceil(fs_n / seq_len)
            seq_len = seq_len * it
            indices = indices * it

        start = random.choice(list(range(0, seq_len - fs_n + 1)))
        end = start + fs_n
        idx_lst = list(range(seq_len))
        idx_lst = idx_lst[start:end]
        idx_lst = sorted(np.random.choice(
            idx_lst, self.seq_length, replace=False))
        indices = [indices[i] for i in idx_lst]
        return [images_path[i] for i in indices], indices


    def __getitem__(self, idx):
        rgb_path = self.all_ids_list_rgb[idx]
        pose_path = self.all_ids_list_pose[idx]
        sil_path = self.all_ids_list_sil[idx]

        rgb_images = natsorted(glob(os.path.join(rgb_path, "*")))
        pose_images = natsorted(glob(os.path.join(pose_path, "*")))
        sil_images = natsorted(glob(os.path.join(sil_path, "*")))

        length_rgb = len(rgb_images)
        length_pose = len(pose_images)
        length_sil = len(sil_images)

        min_length = min(length_rgb, length_pose, length_sil)

        if min_length == length_rgb:
            rgb_data, indices = self.load_images(rgb_images, resize=self.resize, indices=None)
            pose_data, _ = self.load_images(pose_images, resize=self.resize, indices=indices)
            sil_data, _ = self.load_images(sil_images, is_sil=True, resize=self.resize, indices=indices)
        elif min_length == length_pose:
            pose_data, indices = self.load_images(pose_images, resize=self.resize, indices=None)
            rgb_data, _ = self.load_images(rgb_images, resize=self.resize, indices=indices)
            sil_data, _ = self.load_images(sil_images, is_sil=True, resize=self.resize, indices=indices)
        else:
            sil_data, indices = self.load_images(sil_images, is_sil=True, resize=self.resize, indices=None)
            rgb_data, _ = self.load_images(rgb_images, resize=self.resize, indices=indices)
            pose_data, _ = self.load_images(pose_images, resize=self.resize, indices=indices)

        # transform
        all_data = np.ascontiguousarray(np.concatenate([rgb_data, pose_data, sil_data], axis=-1).transpose(0, 3, 1, 2)) # (T, C, H, W)
        all_data = np.ascontiguousarray(self.pixel_transforms(all_data))

        if self.image_finetune:
            rgb_data = numpy_to_pt(all_data[0, :3]) # (T, 3, H, W), (-1, 1)
            pose_data = numpy_to_pt(all_data[0, 3:6]) # (T, 3, H, W), (-1, 1)
            sil_data = numpy_to_pt(all_data[0, 6:], is_sil=True) # (T, 1, H, W), (0-1)
        else:
            rgb_data = numpy_to_pt(all_data[:, :3]) # (T, 3, H, W), (-1, 1)
            pose_data = numpy_to_pt(all_data[:, 3:6]) # (T, 3, H, W), (-1, 1)
            sil_data = numpy_to_pt(all_data[:, 6:], is_sil=True) # (T, 1, H, W), (0-1)
        
        if random.random() < 0.5 :
            new_sil_data = torch.zeros_like(sil_data)
            if len(rgb_data.size()) == 3:
                tmp_mean_value = rgb_data.mean(0, keepdim=True)
            else:
                tmp_mean_value = rgb_data.mean(1, keepdim=True)
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
            text = random.choice(self.text_prompts)
            prompts_list = ["best quality", "photo-realistic", "detailed clothes"]
            text = f"{text}, {random.choice(prompts_list)}"
        else:
            text = "(person, full body), best quality, highres"
        sample = dict(pixel_values=rgb_data, pose_pixel_values=pose_data, fg_pixel_values=fg_data, text=text, sil_pixel_values=sil_data)
        return sample

        

if __name__ == "__main__":

    dataset = GaitLUDatasetImg(
        rgb_root="dataset/train_data/GaitLU_img/train/org",
        pose_root="dataset/train_data/GaitLU_img/train/pose",
        sil_root="dataset/train_data/GaitLU_img/train/sil",
        frame_number=8,
        interval=4,
        sample_size=256,
        image_finetune=False
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(f"========================{idx:06d}========================")
        print(batch["pixel_values"].shape)
        print(batch["pose_pixel_values"].shape)
        print(batch["sil_pixel_values"].shape)
        if idx == 10:
            break
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)