from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8_ppe.pt")

@app.get('/')
async def home():
    print("Hello World")
    return JSONResponse("Hello World")

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    file_name = file.filename
    contents = await file.read()
    with open(file_name, "wb") as f:
        f.write(contents)
    result = model.predict(file_name, save=True, show=True)
    return JSONResponse(result)