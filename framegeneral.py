import os
import sys
import subprocess
 
 
OUT_DATA_DIR="/home/ghl/code/TimeSformer/dataset/video_pics"
txt_path = "/home/ghl/code/TimeSformer/dataset/video.txt"
 
 
filelist = []
i = 1
with open(txt_path, 'r', encoding='utf-8') as f:
  for line in f:
    line = line.rstrip('\n')
    video_name = line.split('\t')[0].split('.')[0]
    dst_path = os.path.join(OUT_DATA_DIR, video_name)
    video_path = line.split('\t')[1]
    if not os.path.exists(dst_path):
      os.makedirs(dst_path)
    print(i)
    i += 1
    cmd = 'ffmpeg -i \"{}\" -r 1 -q:v 2 -f image2 \"{}/%05d.jpg\"'.format(video_path, dst_path)
    subprocess.call(cmd, shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)