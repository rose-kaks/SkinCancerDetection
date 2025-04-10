from flask import Flask, render_template, request, send_from_directory
import os
import onnxruntime as ort
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from datetime import datetime
from werkzeug.utils import secure_filename
import requests
import gdown

model_path = "vit_skin.onnx"
file_id = "1ejawjJL0USWXZoMSv_VEFpkvL5QLxGLc"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    print("Downloading model using gdown...")
    gdown.download(gdrive_url, model_path, quiet=False)
else:
    print("Model already exists.")

# load the model
session = ort.InferenceSession("vit_skin.onnx")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load ONNX model
onnx_model = ort.InferenceSession("vit_skin.onnx")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_skin_disease(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).numpy()

    ort_inputs = {onnx_model.get_inputs()[0].name: image}
    ort_outs = onnx_model.run(None, ort_inputs)
    
    logits = torch.tensor(ort_outs[0])
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred].item() * 100

    return ("nv" if pred == 0 else "mel"), confidence

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    img = None
    
    if request.method == 'POST':
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                label, conf = predict_skin_disease(Image.open(path))
                result = label
                confidence = conf
                img = filename
        
        # Handle webcam image (binary version)
        elif 'webcamImage' in request.files:
            file = request.files['webcamImage']
            if file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"webcam_{timestamp}.jpg"
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the image
                file.save(path)
                
                # Predict
                image = Image.open(path)
                label, conf = predict_skin_disease(image)
                result = label
                confidence = conf
                img = filename
    
    return render_template('index.html', result=result, confidence=confidence, img=img)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)