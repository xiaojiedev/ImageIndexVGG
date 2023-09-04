import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from numpy import linalg as la

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['KERAS_BACKEND'] = 'theano'


class VGGNet:
    # 初始化
    def __init__(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.compat.v1.Session(config=config))
        # 输入图像的大小
        self.input_shape = (224, 224, 3)
        # 池化方式
        self.pooling = 'max'
        # 使用的模型
        # - weights：指定使用哪个预训练模型的权重。该参数可以是None（不使用预训练权重），
        #       'imagenet' 表示使用在ImageNet数据集上预训练的权重
        # - input_shape：指定输入图像的大小
        # - include_top：指定是否包含模型的顶部（即全连接层），False表示不包含顶部，只包含模型的卷积部分。
        self.model = VGG16(weights='imagenet',
                           input_shape=(self.input_shape[0],
                                        self.input_shape[1], self.input_shape[2]), pooling=self.pooling,
                           include_top=False)
        # 模型预测操作,预先加载模型权重，以便在后续使用该模型时能够更快地进行推理
        # 具体来说，这里使用了一个全零数组作为输入，该数组的维度为(1, 224, 224, 3)，
        # 与VGG16模型所需的输入维度相同。使用全零数组作为输入的原因是，
        # 在预测时，模型会根据输入的形状来推断其内部的参数，从而自动初始化权重。
        # 这样，在后续使用该模型时，就可以避免在每次进行预测时都要重新初始化模型权重的问题，从而提高推理速度。
        self.model.predict(np.zeros((1, 224, 224, 3)))

    # 获取最后一层卷积输出的特征
    def extract_feat(self, img_path):
        # 加载指定路径的图像为PIL格式，并将其调整为指定大小
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        # 将图像转换为numpy数组
        img = image.img_to_array(img)
        # 在第0个维度上添加一个维度，将图像转换为4D张量
        img = np.expand_dims(img, axis=0)
        # 对图像进行预处理，将其转换为VGG16模型可以接受的数据格式
        img = preprocess_input(img)
        # 使用VGG16模型对预处理后的图像进行特征提取
        feat = self.model.predict(img)
        # 对提取的特征进行归一化处理，即将其除以特征向量的L2范数
        norm_feat = feat[0] / la.norm(feat[0])
        # 返回处理后的特征向量
        return norm_feat
