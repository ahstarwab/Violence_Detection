import sys
sys.path.append('..')
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
import pickle
import pdb
import cv2
import numpy as np
import torchvision


MEAN_STATISTICS = {
    'imagenet': [0.485, 0.456, 0.406],
    'kinetics': [0.434, 0.405, 0.378],
    'activitynet': [0.450, 0.422, 0.390],
    'none': [0.0, 0.0, 0.0]
}
STD_STATISTICS = {
    'imagenet': [0.229, 0.224, 0.225],
    'kinetics': [0.152, 0.148, 0.157],
    'none': [1.0, 1.0, 1.0]
}


class RWF(Dataset):
    def __init__(self, data_dir, data_partition, clip_len=30, image_size=224, temporal_stride=-1, normalize_mode='video'):
        super(RWF, self).__init__()

        '''
        data_partition = 'train' or 'val'
        clip_len = should be a multiple of 3
        temporal_stride = -1 : 'uniform sampling' // else : 'stride sampling'
        normalize_mode = 'video' or 'imagenet' or 'kinetics'
        '''

        self.data_dir = Path(data_dir)
        self.data_partition = data_partition
        self.video_list = []
        self.labels = []
        self.clip_len = clip_len
        self.temporal_stride = temporal_stride
        self.normalize_mode = normalize_mode

        # Load data list
        i = 0
        for label in ['NonFight', 'Fight']:    
            vid_list = [x for x in self.data_dir.joinpath(label).iterdir() if x.suffix == '.npy']
            self.video_list.extend(vid_list)
            self.labels.extend([i]*len(vid_list))
            i += 1

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        rgb_list = []
        
        video = np.load(self.video_list[idx], mmap_mode='r', allow_pickle=True)
        video = np.float32(video)

        '''stride sampling'''
        if self.data_partition == 'train' and self.temporal_stride != -1:
            data = self.stride_sampling(video, self.clip_len+1, self.temporal_stride)
        else:
            '''uniform sampling'''
            data = self.uniform_sampling(video, self.clip_len+1)

        data[...,:3] = self.color_jitter(data[...,:3])
        data = self.random_flip(data, prob=0.5)

        for image_ in data:
            rgb, _ = np.array_split(image_, 2, axis=-1 )
            rgb_list.append(torch.from_numpy(rgb.transpose(-1,0,1).copy()))
        
        return self.normalize(torch.stack(rgb_list)), self.labels[idx] #[T x C x H x W], scalar

    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video    

    def stride_sampling(self, video, target_frames, stride):
        vid_len = len(video)

        if vid_len >= (target_frames-1)*stride + 1:
            start_idx = np.random.randint(vid_len - (target_frames-1)*stride)
            data = video[start_idx:start_idx+(target_frames-1)*stride+1:stride]
            

        elif vid_len >= target_frames:
            start_idx = np.random.randint(len(video) - target_frames)
            data = video[start_idx:start_idx + target_frames + 1]

        # Need Zero-pad
        else:
            sampled_video = []
            for i in range(0, vid_len):
                sampled_video.append(video[i])

            num_pad = target_frames - len(sampled_video)
            if num_pad>0:
                while num_pad > 0:
                    if num_pad > len(video):
                        padding = [video[i] for i in range(len(video))]
                        sampled_video += padding
                        num_pad -= len(video)
                    else:
                        padding = [video[i] for i in range(num_pad)]
                        sampled_video += padding
                        num_pad = 0
            data = np.array(sampled_video, dtype=np.float32)
        
        return data
        
    def uniform_sampling(self, video, target_frames):
        # get total frames of input video and calculate sampling interval 
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and 
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])     
        # calculate numer of padded frames and fix it 
        num_pad = target_frames - len(sampled_video)
        if num_pad>0:
            padding = [video[i] for i in range(-num_pad,0)]
            sampled_video += padding     
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)
    

    def color_jitter(self,video):
        # range of s-component: 0-1
        # range of v component: 0-255
        s_jitter = np.random.uniform(-0.2,0.2)
        v_jitter = np.random.uniform(-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[...,1] + s_jitter
            v = hsv[...,2] + v_jitter
            s[s<0] = 0
            s[s>1] = 1
            v[v<0] = 0
            v[v>255] = 255
            hsv[...,1] = s
            hsv[...,2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video


    def normalize(self, data):
        if self.normalize_mode == 'video':
            mean = torch.mean(data)
            std = torch.std(data)
            return (data-mean) / std
        else:
            mean = torch.FloatTensor(MEAN_STATISTICS[self.normalize_mode])
            std = torch.FloatTensor(STD_STATISTICS[self.normalize_mode])
            return (data/255.-mean.view(3,1,1)) / std.view(3,1,1)            
