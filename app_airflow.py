from airflow import DAG
import os
from airflow.operators.python import PythonOperator
#from airflow.operators.dummy import DummyOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import numpy as np
import pickle
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow import keras
import cv2
import glob
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

DATA_PATH = 'Airflow/data/*.jpg'
MODEL_PATH = 'Airflow/data/model_rumah123.pkl'    

def load_data_task(data_dir):
    labels = ['non_watermark', 'r123-watermark']
    img_size = 512
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                resized_arr = resized_arr[200:300, 50:450]
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.asarray(data, dtype="object")

def train_model(data_dir):
    train = load_data_task(data_dir)
    x_train = []
    y_train = []
    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    collect_image=[]
    for img in x_train:
        img = cv2.resize(img, (128, 128)) 
        imgFloat = img.astype(float) / 255.
        kChannel = 1 - np.max(imgFloat, axis=2)
        adjustedK = cv2.normalize(kChannel, None, 0, 2.17, cv2.NORM_MINMAX, cv2.CV_32F)
        adjustedK = (255*adjustedK).astype(np.uint8)
        binaryImg = cv2.adaptiveThreshold(adjustedK, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 11)
        textMask = cv2.GaussianBlur(binaryImg, (3, 3), cv2.BORDER_DEFAULT)
        collect_image.append(keras.applications.mobilenet.preprocess_input(textMask))
    x_train = np.array(collect_image)
    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
    y_train = np.array(y_train)
      
    model = pickle.load(open(MODEL_PATH, 'rb'))
    epoch =100
    bacth = 20
    history = model.fit(x_train,y_train, batch_size = bacth ,epochs = epoch)
    pickle.dump(model, open(MODEL_PATH, 'wb'))
    return model

# Define DAG and its default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 1),
    'retries': 1
}

dag = DAG(
    dag_id='ml_pipeline', 
    default_args=default_args, 
    description='Machine Learning Pipeline', 
    schedule='@monthly'
)

# Create tasks for each function
start_task = EmptyOperator(task_id='start', dag=dag)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

end_task = EmptyOperator(task_id='end', dag=dag)

# Set task dependencies
start_task >> train_model_task >> end_task