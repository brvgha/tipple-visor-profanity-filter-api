import tensorflow as tf
import io
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware

# Load the Keras model
image_filter_model = tf.keras.models.load_model("profanify_filter_model.keras")
text_filter_model = tf.keras.models.load_model("text_profanity_model.keras")

with open("./tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
    f.close()

print("loaded vectorizer")
def preprocess_new_image(image, img_size=(128, 128)):
    # convert to tensor
    image = tf.convert_to_tensor(np.array(image, dtype = np.float32), dtype = tf.float32)
    # resize for model
    image = tf.image.resize(image, img_size)
    # normalize to 1 or 0
    image = image / 255.0
    return tf.expand_dims(image, axis=0)

class TextInput(BaseModel):
    text: str

origins = [
    "http://localhost",  # Allow local development for front-end
    "http://localhost:8000",  # FastAPI server itself
    "*"
]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/image_predict")
async def image_predict(file: UploadFile = File(...)):
    print(f"Image received")
    image = Image.open(io.BytesIO(await file.read()))
    processed_input =  preprocess_new_image(image)
    
    # Make prediction
    prediction = image_filter_model.predict(processed_input)

    # Assuming the model outputs a probability, adjust as needed
    is_profane = True if prediction > 0.5 else False
    print(f"Result = {is_profane}")
    return {"statusCode" : 200, "body" : is_profane}

@app.post("/text_predict")
async def text_predict(text: TextInput):
    print(f"Text received: {text}")
    prediction = tfidf.transform([text.text]).toarray()
    prediction = text_filter_model.predict(prediction)
    is_profane = True if prediction > 0.8 else False
    print(f"Result = {is_profane}")
    return {"statusCode" : 200, "body" : is_profane}