from ultralytics import YOLO
from flask import Flask, request, jsonify, flash, redirect, url_for, render_template
from flask_cors import CORS, cross_origin
import base64
from PIL import Image
from io import BytesIO
import json
import shutil

app = Flask(__name__)
CORS(app)
model = YOLO("yolov8_ppe.pt")

try:
    shutil.rmtree("runs/detect/predict")
except:
    pass

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

@app.route('/')
def home():
    print("Hello World")
    return jsonify("Hello World")

@app.route('/predict', methods=['POST'])
def predict():
   # return jsonify(request.form)
    if request.method == 'POST':
        if 'image' in request.json:
            imagen_base64 = request.json['image']
            imagen_bytes = base64.b64decode(imagen_base64)
            imagen = Image.open(BytesIO(imagen_bytes))
            imagen.save("imagen.jpg")
            #wreturn jsonify(imagen_base64)
            if imagen is None:
                return jsonify("No file found")
            else:
                model.predict("imagen.jpg", save=True )
                image_path = "runs/detect/predict/imagen.jpg"
                base64_image = image_to_base64(image_path)
                return jsonify(base64_image)
        
if __name__ == '__main__':
    #app.run(debug=True, port=8000)
    app.run(debug=True, port=8000, host="172.31.27.125")


