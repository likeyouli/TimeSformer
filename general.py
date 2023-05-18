import os 
import csv
import shutil
#from tqdm import tqdm
from sklearn.model_selection import train_test_split

out_dir = "/home/ghl/code/TimeSformer/result"  # 全路径
video_path = "/home/ghl/code/TimeSformer/dataset/UCF-101" # 全路径MCQ/data/MSRVTT/vodeos/all
file_name = ".csv"
video_name = ".avi"
name_list = ["train","test","val"]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, 'train'))
    os.mkdir(os.path.join(out_dir, 'val'))
    os.mkdir(os.path.join(out_dir, 'test'))
for file in os.listdir(video_path):
        file_path = os.path.join(video_path, file)
        print("-------",len(file_path))
        video_files = [name for name in os.listdir(file_path)]
        #print('----datatype:----',type(video_files))
        
        train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
        train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
        train_dir = os.path.join(out_dir, 'train', file)
        val_dir = os.path.join(out_dir, 'val', file)
        test_dir = os.path.join(out_dir, 'test', file)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        '''for video in tqdm(train):
           shutil.copy(os.path.join(video_path,file,video),os.path.join(train_dir,video))
        for video in tqdm(test):
            shutil.copy(os.path.join(video_path,file,video),os.path.join(test_dir,video))
        for video in tqdm(val):
            shutil.copy(os.path.join(video_path,file,video),os.path.join(val_dir,video))'''
if not os.path.exists(os.path.join(out_dir,"csv")):
    os.mkdir(os.path.join(out_dir,"csv"))
    for name in name_list:
        with open(os.path.join(out_dir,"csv",name+file_name),'wb') as f:
            print("创建"+os.path.join(out_dir,"csv",name+file_name))
csv_path = os.path.join(out_dir,"csv")
for ii in os.listdir(path=csv_path):
    if ii.split(".")[0] in name_list:
        path1 = os.path.join(csv_path,ii)
        with open(path1, 'w', newline='') as f:
            for dd in os.listdir(out_dir):
                if dd==ii.split(".")[0]:
                    for zz in os.listdir(os.path.join(out_dir,dd)):
                        for mm in os.listdir(os.path.join(out_dir,dd,zz)):
                            writer = csv.writer(f)
                            writer.writerow([os.path.join(out_dir,dd,zz,mm),zz]) 

## 创建类别label标号文件
labels= []
for label in sorted(os.listdir(video_path)):
    labels.append(label)
label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
label_file = os.path.join(out_dir, str(len(os.listdir(video_path))) + 'class_labels.txt')
with open(label_file, 'w') as f:
    for id, label in enumerate(sorted(label2index)):
        f.writelines(str(id) + ' ' + label + '\n')
#替换csv文件中类别名为数字
csv_file = os.path.join(out_dir,"csv")
def txt_read(files):
    txt_dict = {}
    fopen = open(files)
    for line in fopen.readlines():
        line = str(line).replace("\n","") #注意，必须是双引号，找了大半个小时，发现是这个问题。。
        txt_dict[line.split(' ',1)[1]] = line.split(' ',1)[0]
        #split（）函数用法，逗号前面是以什么来分割，后面是分割成n+1个部分，且以数组形式从0开始
        #初学python，感觉这样表达会理解一点。。
    fopen.close()
    return txt_dict

txt_dict = txt_read(label_file)
# print(txt_dict)

for ii in os.listdir(csv_file):
    path1 = os.path.join(csv_file,ii)
    r = csv.reader(open(path1))
    lines = [l for l in r]
    for i in range(len(lines)):
        cs = lines[i][1]
        value = txt_dict[cs]
        lines[i][1] = value
    writer = csv.writer(open(path1, 'w'))
    writer.writerows(lines)
