from flask import Flask, json, request, jsonify, abort
import requests
import time
import os

from model_loader import model_loader

app = Flask(__name__)

app.debug = True

counsellor = model_loader()

@app.route('/AI/sendMessage/',methods=['POST'])
def counsel():
    global counsellor

    body = request.get_json()
    message = body['message']
    message = counsellor.split_msg(message)

    emotion = counsellor.classify_msg(message)
    words = counsellor.tokenize_msg(message)

    return emotion, words


if __name__ == '__main__':
    app.run(debug=True)