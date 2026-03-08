from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from

app = Flask(__name__)

# 启用 Flasgger，自动提供 /apidocs 路径
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

Swagger(app, config=swagger_config)

# 模拟数据库：内存字典
students = {
    1: {"name": "张三", "age": 20, "grade": "大二"},
    2: {"name": "李四", "age": 19, "grade": "大一"}
}
next_id = 3


@app.route('/students', methods=['GET'])
@swag_from({
    'tags': ['学生管理'],
    'summary': '获取所有学生信息',
    'description': '返回当前所有学生的列表（以学号为键）',
    'responses': {
        200: {
            'description': '成功返回学生列表',
            'schema': {
                'type': 'object',
                'additionalProperties': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'age': {'type': 'integer'},
                        'grade': {'type': 'string'}
                    }
                }
            }
        }
    }
})
def get_all_students():
    return jsonify(students)


@app.route('/students/<int:student_id>', methods=['GET'])
@swag_from({
    'tags': ['学生管理'],
    'summary': '根据ID获取单个学生',
    'parameters': [
        {
            'name': 'student_id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': '学生的唯一ID'
        }
    ],
    'responses': {
        200: {
            'description': '成功返回学生信息',
            'schema': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'integer'},
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                    'grade': {'type': 'string'}
                }
            }
        },
        404: {'description': '学生未找到'}
    }
})
def get_student(student_id):
    student = students.get(student_id)
    if student:
        return jsonify({"id": student_id, **student})
    else:
        return jsonify({"error": "学生未找到"}), 404


@app.route('/students', methods=['POST'])
@swag_from({
    'tags': ['学生管理'],
    'summary': '新增一个学生',
    'consumes': 'application/json',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'example': '王五'},
                    'age': {'type': 'integer', 'example': 21},
                    'grade': {'type': 'string', 'example': '大三'}
                },
                'required': ['name', 'age', 'grade']
            }
        }
    ],
    'responses': {
        201: {
            'description': '学生创建成功',
            'schema': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'integer'},
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                    'grade': {'type': 'string'}
                }
            }
        },
        400: {'description': '请求数据不完整'}
    }
})
def add_student():
    global next_id
    data = request.get_json()

    if not data or not all(k in data for k in ('name', 'age', 'grade')):
        return jsonify({"error": "缺少必要字段：name, age, grade"}), 400

    students[next_id] = {
        "name": data['name'],
        "age": data['age'],
        "grade": data['grade']
    }
    result = {"id": next_id, **students[next_id]}
    next_id += 1
    return jsonify(result), 201


@app.route('/students/<int:student_id>', methods=['PUT'])
@swag_from({
    'tags': ['学生管理'],
    'summary': '更新学生信息',
    'parameters': [
        {
            'name': 'student_id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': '要更新的学生ID'
        },
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'example': '赵六'},
                    'age': {'type': 'integer', 'example': 22},
                    'grade': {'type': 'string', 'example': '大四'}
                }
            }
        }
    ],
    'responses': {
        200: {
            'description': '更新成功',
            'schema': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'integer'},
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                    'grade': {'type': 'string'}
                }
            }
        },
        404: {'description': '学生未找到'},
        400: {'description': '请求体为空'}
    }
})
def update_student(student_id):
    if student_id not in students:
        return jsonify({"error": "学生未找到"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    if 'name' in data:
        students[student_id]['name'] = data['name']
    if 'age' in data:
        students[student_id]['age'] = data['age']
    if 'grade' in data:
        students[student_id]['grade'] = data['grade']

    return jsonify({"id": student_id, **students[student_id]})


@app.route('/students/<int:student_id>', methods=['DELETE'])
@swag_from({
    'tags': ['学生管理'],
    'summary': '删除一个学生',
    'parameters': [
        {
            'name': 'student_id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': '要删除的学生ID'
        }
    ],
    'responses': {
        200: {'description': '删除成功，返回确认消息'},
        404: {'description': '学生未找到'}
    }
})
def delete_student(student_id):
    if student_id not in students:
        return jsonify({"error": "学生未找到"}), 404

    del students[student_id]
    return jsonify({"message": f"学生 {student_id} 已删除"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)