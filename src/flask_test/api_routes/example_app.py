# api_routes/example_app.py
from flask import request, jsonify
from flasgger import swag_from
import os

def add_example_routes(app):
    """
    向 Flask app 动态添加路由
    注意：这里仍然可以用 @swag_from，只要 Swagger 已初始化
    """

    @app.route('/api/v1/hello', methods=['GET'])
    @swag_from({
        'tags': ['示例接口'],
        'summary': '打招呼接口',
        'description': '返回一句问候语',
        'parameters': [
            {
                'name': 'name',
                'in': 'query',
                'type': 'string',
                'required': False,
                'description': '你的名字'
            }
        ],
        'responses': {
            200: {
                'description': '成功返回问候语',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {'type': 'string'}
                    }
                }
            }
        }
    })
    def hello():
        name = request.args.get('name', 'World')
        return jsonify({"message": f"Hello, {name}!"})

    @app.route('/api/v1/data', methods=['POST'])
    @swag_from({
        'tags': ['示例接口'],
        'summary': '接收数据并回显',
        'consumes': 'application/json',
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'required': True,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'integer', 'example': 123},
                        'content': {'type': 'string', 'example': 'test data'}
                    },
                    'required': ['id', 'content']
                }
            }
        ],
        'responses': {
            200: {
                'description': '成功回显',
                'schema': {'type': 'object'}
            }
        }
    })
    def echo_data():
        data = request.get_json()
        return jsonify({"received": data})