import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
#from timesformer.absFT.dataloader import *
from timesformer.absFT.dataloader import VideoClassificationDataset
#TimeSformer.absFT.
from timesformer.models.vit import TimeSformer
 
device = torch.device("cuda:0")
 
if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('test_list_dir', help="Directory where test features are stored.")
    
    opt = vars(opt.parse_args())
 
    test_opts = {'feats_dir': opt['test_list_dir']}
 
    # =================模型建立======================
    model = TimeSformer(img_size=224, num_classes=10000, num_frames=8, attention_type='divided_space_time',
                        pretrained_model='/home/ghl/code/TimeSformer/TimeSformer_divST_8x32_224_HowTo100M.pyth')
 
    model = model.eval().to(device)
    print(model)
 
    # ================数据加载========================
    print("Use", torch.cuda.device_count(), 'gpus')
    test_loader = {}
 
    test_dataset = VideoClassificationDataset(test_opts, 'val')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=6, shuffle=False)
 
    # ===================训练和验证========================
    i = 0
    file1 = open("/home/ghl/code/TimeSformer/dataset/video_pics.txt", 'r')#/home/ghl/code/TimeSformer/
    file1_list = file1.readlines()
    for data in test_loader:
        model_input = data['fc_feats'].to(device)
        name_feature = file1_list[i].rstrip().split('\t')[0].split('.')[0]
        i = i + 1
        out = model(model_input, )
        out = out.squeeze(0)
        out = out.cpu().detach().numpy()
        np.save('/home/ghl/code/TimeSformer/video_feature/' + name_feature + '.npy', out)

        print(i)



