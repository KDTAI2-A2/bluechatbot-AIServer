from flask import Flask, request, jsonify

from model_loader import model_loader

import time

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

counsellor = model_loader()

@app.route('/AI/sendMessage/',methods=['POST'])
def counsel():
    start = time.time()  # Response Time Check

    global counsellor

    message = request.get_json()
    #message = counsellor.split_msg(message)

    emotion = counsellor.classify_msg(message)  # 감정 분류 결과
    words = counsellor.tokenize_msg(message)    # 단어 추출 결과
    reply = counsellor.generate_reply(message)  # 답변 생성 결과

    response = jsonify(
        emotion,
        words,
        reply
    )

    print(f"Response time : {time.time() - start}")  # Response Time Check

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)