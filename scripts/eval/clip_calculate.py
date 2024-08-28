"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could 
measure the similarity of cross modalities. Please find more information from 
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate 
the mean average of cosine similarities. 

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from kn_util.utils import setup_logger_loguru, setup_debugpy

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
from einops import rearrange, repeat
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

TEXT_EXTENSIONS = {'txt'}


class DummyDataset(Dataset):
    
    FLAGS = ['img', 'txt', 'vid']
    def __init__(self, vid_path, txt_path, frame = 16,
                 transform = None,
                 tokenizer = None) -> None:
        super().__init__()
        self.vid_folder = self._combine_without_prefix(vid_path)
        self.txt_content = self._load_txt_prompt(txt_path)
        self.transform = transform
        self.tokenizer = tokenizer
        # assert self._check()



    def __len__(self):
        return len(self.vid_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        vid_path = self.vid_folder[index]
        vid_data = self._load_vid(vid_path)

        sample = dict(vid=vid_data, txt=self._process_txt(self.txt_content[index]))
        return sample
    
    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_vid(self,path):
        reader = imageio.get_reader(path,'ffmpeg')
        vid = [ Image.fromarray(_) for _ in reader]
        if self.transform is not None:
            vid = [torch.unsqueeze(self.transform(_),dim=0) for _ in vid]
        # import ipdb; ipdb.set_trace()
        return torch.concat(vid,dim =0 )


    def _load_img(self, path):
        img = Image.open(path)
        
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _process_txt(self,txt):
        if self.tokenizer is not None:
            txt = self.tokenizer(txt).squeeze()
        return txt
    
    def _load_txt(self, path):
        with open(path, 'r') as fp:
            data = fp.read()
            fp.close()
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.vid_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True
    
    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder

    def _load_txt_prompt(self,txt_path):
        prompt = []
        with open(txt_path, 'r') as file:
            for line in file:
                prompt.append(line[:-1])
        prompt.sort()
        return prompt


@torch.no_grad()
def calculate_clip_score_simple(dataloader, model):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    for batch_data in tqdm(dataloader):
        vid = batch_data['vid']
        f = vid.shape[1]
        vid = rearrange(vid, "b f c h w -> (b f) c h w")
        vid_features = forward_modality(model, vid, 'img')
        txt = batch_data['txt']
        txt = repeat(txt, "b c -> (b f) c",f =f)
        txt_features = forward_modality(model, txt, 'txt')
        
        # normalize features
        vid_features = vid_features / vid_features.norm(dim=1, keepdim=True).to(torch.float32)
        txt_features = txt_features / txt_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        # calculate scores
        # score = logit_scale * vid_features @ txt_features.t()
        # score_acc += torch.diag(score).sum()
        score = logit_scale * (txt_features * vid_features).sum()
        import ipdb; ipdb.set_trace()
        score_acc += score
        sample_num += vid_features.shape[0]
    
    return score_acc / sample_num

        
def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=2,
                    help='Batch size to use')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                    help='CLIP model to use')
    parser.add_argument('--num-workers', type=int, default=8,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--txt_path', type=str,default= "/home/aiops/zhangfz/base_finetune/Memory/prompts/mira_short_prompt.txt")
    parser.add_argument('--vid_path', type=str, default= "/home/aiops/zhangfz/base_finetune/Memory/mira_video_100")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    print('Loading CLIP model: {}'.format(args.clip_model))
    model, preprocess = clip.load(args.clip_model, device=device)
    
    dataset = DummyDataset(txt_path = args.txt_path, vid_path = args.vid_path,
                           transform=preprocess, tokenizer=clip.tokenize)
    temp = dataset.__getitem__(0)
    # import ipdb; ipdb.set_trace()
    dataloader = DataLoader(dataset, args.batch_size, 
                            num_workers=num_workers, pin_memory=True)
    
    print('Calculating CLIP Score:')
    clip_score = calculate_clip_score_simple(dataloader, model,)
    clip_score = clip_score.cpu().item()
    print('CLIP Score: ', clip_score)


if __name__ == '__main__':
    #setup_debugpy(force=True)
    main()
