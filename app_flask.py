from ultralytics import YOLO
from flask import Flask, request, jsonify, flash, redirect, url_for, render_template

app = Flask(__name__)
model = YOLO("yolov8_ppe.pt")

@app.route('/')
def home():
    print("Hello World")
    return jsonify("Hello World")

@app.route('/predict', methods=['POST'])
def predict():
   # return jsonify(request.form)
    if request.method == 'POST':
        content = request.json
        if content is None:
            return jsonify("No file found")
        else:
            return jsonify(content)
            file = request.files['file']
            file.save(file.filename)
            result = model.predict(file.filename, save=True, show=True)
            return jsonify(result)
        
if __name__ == '__main__':
    app.run(debug=True, port=8000)
    #app.run(debug=True, port=8000, host="172.31.27.125")
