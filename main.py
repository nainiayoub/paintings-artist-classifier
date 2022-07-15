import uvicorn
from fastapi import FastAPI
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import imageio
import cv2
import urllib.request
import os

app = FastAPI()

# urll = 'https://github.com/nainiayoub/paintings-artist-classifier/releases/download/v1.0.0/artists_classifier.h5'
# filename_model = urll.split('/')[-1]
# urllib.request.urlretrieve(urll, filename_model)

# model_file = filename_model
model_file = './models/artists_classifier.h5'


@app.post("/predict-artist/")
def get_image(url: str):
    # arguments
    train_input_shape = (224, 224, 3)
    labels = ['Vincent_van_Gogh', 'Edgar_Degas', 'Pablo_Picasso',
        'Pierre-Auguste_Renoir', 'Albrecht_Dürer', 'Paul_Gauguin',
        'Francisco_Goya', 'Rembrandt', 'Alfred_Sisley', 'Titian',
        'Marc_Chagall']

    # image processing
    web_image = imageio.imread(url)
    web_image = cv2.resize(web_image, dsize=train_input_shape[0:2], )
    web_image = img_to_array(web_image)
    web_image /= 255.
    web_image = np.expand_dims(web_image, axis=0)

    # artist classification
    model = load_model(model_file)
    prediction = model.predict(web_image)
    prediction_probability = np.amax(prediction)
    prediction_idx = np.argmax(prediction)

    return {"name": labels[prediction_idx]}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    # ngrok_tunnel = ngrok.connect(8000)
    # print('Public URL:', ngrok_tunnel.public_url)
    # nest_asyncio.apply()
    # uvicorn.run(app, port=8000)