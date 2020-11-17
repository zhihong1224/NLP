"""
this file is used to test API_URL.
"""

import requests
import flask
import json
import argparse

PyTorch_REST_API_URL = "http://127.0.0.1:5000/predict"

# from the terminal to get input
# def predict_result(sentence):
#     payload = {'file':sentence}
#     r=requests.post("http://127.0.0.1:5000/predict",data=payload).json()
#     # print(r)
#     if r['success']:
#         result=r["predictions"][0]
#         print("input sentence:{},translation sentence:{}".format(result['input sentence'],result['translation sentence']))
#     else:
#         print("Request failed")

# from html to get inpuy
def predict_result():
    sentence=flask.request.form.get('data_input')
    print(sentence)


if __name__ == '__main__':
    # from thr terminal to get input
    # parser = argparse.ArgumentParser(description="Translation demo")
    # parser.add_argument('--file',type=str,help='input test sentence')
    # args=parser.parse_args()
    # predict_result(args.file)

    # from html get input
    predict_result()


