from extract_cnn_vgg16_keras import VGGNet
import numpy as np
import h5py
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-query", required=True,
                help="指定要查询的图像文件")
ap.add_argument("-index", required=True,
                help="指定h5特征数据库")
ap.add_argument("-result", required=True,
                help="指定要匹配的图像文件库")
args = vars(ap.parse_args())

h5f = h5py.File(args["index"], 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

print("----------------------------------------")
print("               检索 开始")
print("----------------------------------------")

# 载入VGG模型
model = VGGNet()

# 获取图像的特征参数
queryDir = args["query"]
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

maxres = 10
imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
print("前 %d 个近似的图像: " % maxres, imlist)
print("相似比: ", rank_score[0:10])
