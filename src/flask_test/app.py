# app.py
from flask import Flask, jsonify
from flasgger import Swagger, swag_from
from route_manager import route_manager

app = Flask(__name__)

# 初始化 Swagger
swagger = Swagger(app, config={
    "headers": [],
    "specs": [{
        "endpoint": 'apispec',
        "route": '/apispec.json',
        "rule_filter": lambda rule: True,
        "model_filter": lambda tag: True,
    }],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
})

# 初始化路由管理器
route_manager.init_app(app)
route_manager.register_blueprints()

@app.route(rule="/get_student", methods=['GET'])
def get_student():
    """
    Get the current version of the application.
    ---
    tags:
      - System
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Version retrieved successfully.
        schema:
          type: object
          properties:
            version:
              type: string
              description: Version number.
    """
    return jsonify({'name': 'zhangsa', 'age': 1})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)