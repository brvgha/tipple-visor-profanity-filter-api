import tensorflow as tf
import io
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile

# Load the Keras model
model = tf.keras.models.load_model("profanify_filter_model.keras")


def preprocess_new_image(image, img_size=(128, 128)):
    # convert to tensor
    image = tf.convert_to_tensor(np.array(image, dtype = np.float32), dtype = tf.float32)
    # resize for model
    image = tf.image.resize(image, img_size)
    # normalize to 1 or 0
    image = image / 255.0
    return tf.expand_dims(image, axis=0)

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    processed_input =  preprocess_new_image(image)
    
    # Make prediction
    prediction = model.predict(processed_input)

    # Assuming the model outputs a probability, adjust as needed
    is_profane = True if prediction > 0.5 else False

    return is_profane