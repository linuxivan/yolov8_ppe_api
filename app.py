from ultralytics import YOLO
from fastapi import FastAPI
app = FastAPI()
model = YOLO("yolov8_ppe.pt")

#result = model.predict("1.jpg", save=True, show=True)

@app.get("/")
def read_root():
    return {"Hello": "World"}