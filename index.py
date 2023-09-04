import argparse
import os

import h5py
import numpy as np

# import sys
# reload(sys)
# sys.setdefaultencoding("utf8")
from extract_cnn_vgg16_keras import VGGNet

ap = argparse.ArgumentParser()
ap.add_argument("-database", required=True,
                help="指定包含图像的文件夹")
ap.add_argument("-index", required=True,
                help="指定生成的H5特征库名")
args = vars(ap.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 需要处理的图像数据目录
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path)
            if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]


# 获取图像特征，生成h5索引文件
if __name__ == "__main__":
    db = args["database"]
    img_list = get_imlist(db)

    print("--------------------------------------------------")
    print("         特征识别&导出 开始")
    print("--------------------------------------------------")

    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))

    feats = np.array(feats)
    output = args["index"]

    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)

    # 图片名中有特殊字符，中文等
    names = [name.encode() for name in names]
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
