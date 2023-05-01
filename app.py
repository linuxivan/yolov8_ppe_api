from ultralytics import YOLO
from flask import Flask, request, jsonify, flash, redirect, url_for, render_template
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
model = YOLO("yolov8_ppe.pt")

@app.route('/')
def home():
    print("Hello World")
    return jsonify("Hello World")

@app.route('/predict', methods=['POST'])
def predict():
   # return jsonify(request.form)
    if request.method == 'POST':
        if 'test' in request.form:
            return jsonify("Test")
        if 'file' not in request.files:
            return jsonify("No file found")
        else:
            file = request.files['file']
            file.save(file.filename)
            result = model.predict(file.filename, save=True, show=True)
            return jsonify(result)
        
if __name__ == '__main__':
    app.run(debug=True, port=8000)
    
