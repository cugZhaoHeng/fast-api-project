from flask import Flask, Response, stream_with_context,jsonify
from flask_cors import CORS  # 新增
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 生成事件流的函数
def generate_numbers():
    count = 0
    while True:
        count += 1
        yield f"data: {count}\n\n"
        time.sleep(1)

@app.route('/source-event')
def source_event():
    def generate():
        count = 0
        while True:
            count += 1
            yield f"data: {count}\n\n"
            time.sleep(1)

    # resp = Response(stream_with_context(generate()), mimetype='text/event-stream')
    # resp.headers['Access-Control-Allow-Origin'] = '*'
    # resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    # resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    # resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    # resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    # resp.headers.add_header("Cache-control", "no-cache")
    # resp.headers.add_header("Connection", "keep-alive")
    # resp.headers.add_header("X-Accel-Buffering", "no")
    # resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp

@app.route("/generate_stream", methods=["GET"])
def genereate_stream():
    def generate():
        with app.test_client() as client:
            for i in range(10):
                yield f"data: {i}\n\n"
                time.sleep(1)
    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        'message': 'Hello, this is a JSON response!',
        'status': 'success',
        'data': {
            'id': 1,
            'name': 'Test'
        }
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6010, debug=True)