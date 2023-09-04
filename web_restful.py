import os

from flask import Flask
from flask_restful import reqparse, Api, Resource
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from query_api import get_image_search

# 创建 Flask 应用和 Flask-RESTful Api 对象
app = Flask(__name__)
CORS(app, resources=r'/*')
api = Api(app)

# 字典RE，定义API 信息和警告信息
RE = {
    "get": {"Doc": "Unipower Image Search System Restful Api"},
    "WARN": {"WARNING": "Null"}
}

# 解析 POST 请求中的文件数据
parser = reqparse.RequestParser()
parser.add_argument('image', type=FileStorage, location='files')


# 处理API请求，继承自 Resource 类
class ImageSearch(Resource):
    def get(self):
        return RE['get'], 200

    def post(self):
        # 解析 POST 请求中的文件数据，获取上传的图片文件对象
        args = parser.parse_args()
        im_file = args.get('image')

        # 文件名安全处理，保存到指定的目录中
        im_name = secure_filename(im_file.filename)
        im_file.save(os.path.join('image/', im_name))

        # 根据图片路径调用方法进行图片搜索
        im_file = os.path.join('image/', im_name)
        result_dict = get_image_search(im_file)

        # 判断搜索结果是否为空
        if result_dict:
            return result_dict, 201
        else:
            return RE['WARN'], 404


# 将 ImageSearch 类注册到 Flask-RESTful Api 对象中，并指定路由为 /imagesearch
api.add_resource(ImageSearch, '/imagesearch')

# 启动服务器
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9393, debug=False)
