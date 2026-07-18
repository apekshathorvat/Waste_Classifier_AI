import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


model = tf.keras.models.load_model(
    "models/waste_classifier.keras"
)


classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


def predict_waste(img_path):

    img = image.load_img(
        img_path,
        target_size=(224,224)
    )


    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    img_array = img_array / 255.0


    prediction = model.predict(img_array)


    index = np.argmax(prediction)


    confidence = round(
        np.max(prediction)*100,
        2
    )


    return f"{classes[index]} ({confidence}%)"