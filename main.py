import uvicorn
from fastapi import FastAPI
import numpy as np
from PIL import Image
# from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import imageio
import cv2
import urllib.request
import os

app = FastAPI()

model_file = './models/model_reduced.tflite'
def load_tflite_model(model_file):
  interpreter = tf.lite.Interpreter(model_path = model_file)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model(model_file)


@app.post("/predict-artist/")
def get_image(url: str):
    # arguments
    input_shape = input_details[0]['shape']

    # preprocessing url  
    train_input_shape = (224, 224, 3)
    web_image = imageio.imread(url)
    web_image = cv2.resize(web_image, dsize=train_input_shape[0:2], )
    web_image = img_to_array(web_image)
    web_image /= 255.
    web_image = np.expand_dims(web_image, axis=0)  
    interpreter.set_tensor(input_details[0]['index'], web_image)
    interpreter.invoke()

    labels = ['Vincent_van_Gogh', 'Edgar_Degas', 'Pablo_Picasso',
        'Pierre-Auguste_Renoir', 'Albrecht_Dürer', 'Paul_Gauguin',
        'Francisco_Goya', 'Rembrandt', 'Alfred_Sisley', 'Titian',
        'Marc_Chagall']

    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])   
    prediction_probability = np.amax(probabilities) 
    prediction_idx = np.argmax(probabilities)

    return {"name": labels[prediction_idx]}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    # ngrok_tunnel = ngrok.connect(8000)
    # print('Public URL:', ngrok_tunnel.public_url)
    # nest_asyncio.apply()
    # uvicorn.run(app, port=8000)