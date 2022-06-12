from fastapi import FastAPI, File, UploadFile, Form
from keras.preprocessing import image
from keras.saving.saved_model.load import load
from skimage import io
import uvicorn
import cv2
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from typing import List
from starlette.responses import RedirectResponse

app = FastAPI(debug=True)

MODEL = tf.keras.models.load_model("app/my_model/transfer_model (1).h5")

@app.post("/predict")
async def predict(url: str = Form()):
    image = io.imread(url);
    img = cv2.resize(image,(224,224)) 
    img = img.reshape(224,224,3) 

    img_batch = np.expand_dims(img, 0)
  
    predictions = MODEL.predict(img_batch)
    confidence = np.max(predictions[0])
    if confidence >= 0.001 :
      prediction = 'Kertas'
      print(prediction)
    else:
      prediction = 'Buku'
      print(prediction)

    return {
        'class': prediction,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run("app", port=8000, reload=True)