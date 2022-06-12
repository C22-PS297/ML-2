from fastapi import FastAPI, File, UploadFile
from keras.preprocessing import image
from keras.saving.saved_model.load import load
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

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
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
        'name': file.filename,
        'type':file.content_type,
        'class': prediction,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run("app", port=8000, reload=True)