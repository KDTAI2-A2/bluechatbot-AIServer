from flask import Flask, json, request, jsonify, abort
import time
import os

from model_loader import model_loader

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

counsellor = model_loader()

@app.route('/AI/sendMessage/',methods=['POST'])
def counsel():
    global counsellor

    message = request.get_json()
    #message = counsellor.split_msg(message)

    emotion = counsellor.classify_msg(message)  # 감정 분류 결과
    words = counsellor.tokenize_msg(message)    # 단어 추출 결과
    reply = counsellor.generate_reply(message)

    response = jsonify(
        emotion,
        words,
        reply
    )

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)