import h5py
import numpy as np
import tensorflow as tf

from extract_cnn_vgg16_keras import VGGNet


def get_image_search(im_file):
    # 指定h5库
    h5f = h5py.File('cifar10.h5', 'r')
    # h5文件中的特征向量
    feats = h5f['dataset_1'][:]
    # h5文件中的图像名称
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    model = VGGNet()
    q_vector = model.extract_feat(im_file)
    tf.compat.v1.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    scores = np.dot(q_vector, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    maxres = 10
    im_list = [str(imgNames[index].decode()) for i, index in enumerate(rank_ID[0:maxres])]
    im_score = [str(rank_score[i]) for i in range(maxres)]
    result_dict = dict(zip(im_list, im_score))
    return result_dict
