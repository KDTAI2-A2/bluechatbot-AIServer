from flask import Flask, json, request, jsonify, abort
import time
import os

from model_loader import model_loader

app = Flask(__name__)

#app.debug = True

counsellor = model_loader()

@app.route('/AI/sendMessage/',methods=['POST'])
def counsel():
    global counsellor

    message = request.get_json()
    #message = counsellor.split_msg(message)
    print(message)

    emotion = counsellor.classify_msg(message)
    words = counsellor.tokenize_msg(message)

    return emotion, words


if __name__ == '__main__':
    app.run(debug=True)