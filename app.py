import os
import numpy as np
from util import base64_to_pil
import pickle
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow import keras
import cv2
import os

from keras.preprocessing import image
from PIL import Image
import glob
    


Model_json = "model_rumah123.pkl"
Model_weigths = "model_rumah123.h5"


# Declare a flask app
app = Flask(__name__)

def get_ImageClassifierModel():
    model = pickle.load(open('model_rumah123.pkl', 'rb'))
    return model  
    


def model_predict(img, model):
    img = img.convert('RGB')
    img = np.array(img)
    resized_arr = cv2.resize(img, (512, 512))
    resized_arr = resized_arr[200:300, 50:450]
    img = cv2.resize(resized_arr, (128, 128)) 
    imgFloat = img.astype(float) / 255.
    kChannel = 1 - np.max(imgFloat, axis=2)
    adjustedK = cv2.normalize(kChannel, None, 0, 2.17, cv2.NORM_MINMAX, cv2.CV_32F)
    adjustedK = (255*adjustedK).astype(np.uint8)
    binaryImg = cv2.adaptiveThreshold(adjustedK, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 11)
    textMask = cv2.GaussianBlur(binaryImg, (3, 3), cv2.BORDER_DEFAULT)
    textMask  = keras.applications.mobilenet.preprocess_input(textMask)
    textMask = textMask.reshape(1,128, 128, 1)
    preds = model.predict(textMask)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        model = get_ImageClassifierModel()
        preds = model_predict(img, model)
        pred_proba = "{:.3f}".format(np.amax(preds))
        pred_class = ['r123-watermark' if np.round(preds)==1 else 'no_watermark']
        result = pred_class[0]
        return jsonify(result=result, probability=pred_proba)
    return None


if __name__ == '__main__':
    # app.run(port=5002)
    app.run()
