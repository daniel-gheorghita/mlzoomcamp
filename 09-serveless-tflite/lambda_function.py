# TF Lite prediction without TF/Keras dependencies

from PIL import Image
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
import requests
from io import BytesIO
import numpy as np

classes = ['dino', 'dragon']

def preprocess_input(x):
    x /= 255
    return x

def image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((150,150), Image.Resampling.NEAREST)
    return img

def image_from_file(filepath):
    with Image.open(filepath) as img:
        img = img.resize((150,150), Image.Resampling.NEAREST)
    return img

url_with_dino = "https://i.natgeofe.com/n/9b87dda3-9946-4a1c-a97f-c21f73ced888/Meraxes-CREDIT-Carlos-Papolio_square.jpg"
img_url = image_from_url(url_with_dino)
#img_dino = image_from_file("dino.jpg")
#img_dragon = image_from_file("dragon.jpg")
#img_url

x_url = np.array(img_url, dtype=np.float32)
#x_dino = np.array(img_dino, dtype=np.float32)
#x_dragon = np.array(img_dragon, dtype=np.float32)

#X = np.array([x_url, x_dino, x_dragon])
#X = preprocess_input(X)
#X.shape[0]


# Open the TF Lite model
tflite_interpreter = tflite.Interpreter("dino_dragon_v1.tflite")
input_index = tflite_interpreter.get_input_details()[0]["index"]
output_index = tflite_interpreter.get_output_details()[0]["index"]

def predict(url):
    # Get image from URL
    #img_url = image_from_url(url_with_dino)
    img_url = image_from_url(url)

    x_url = np.array(img_url, dtype=np.float32)
    X = np.array([x_url])
    X = preprocess_input(X)

    # In case of batching
    tflite_interpreter.resize_tensor_input(input_index, [X.shape[0], 150, 150, 3])
    tflite_interpreter.allocate_tensors()
    # Set input to the TF Lite model
    tflite_interpreter.set_tensor(input_index, X)
    # Predict
    tflite_interpreter.invoke()
    # Get prediction output
    pred = tflite_interpreter.get_tensor(output_index)
    pred = pred.flatten().tolist()
    #print(pred.shape)
    #print([classes[np.round(prediction[0]).astype(np.uint8)] for prediction in pred])
    return dict(zip([classes[np.round(prediction).astype(np.uint8)] for prediction in pred], pred))

def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result

#print(predict(url_with_dino))