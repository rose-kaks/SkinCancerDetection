from flask import Flask, render_template, request, send_from_directory
import os
import onnxruntime as ort
from PIL import Image
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Download model if not present
model_path = "vit_skin.onnx"
file_id = "1ejawjJL0USWXZoMSv_VEFpkvL5QLxGLc"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    print("Downloading model using gdown...")
    gdown.download(gdrive_url, model_path, quiet=False)
else:
    print("Model already exists.")

# Load ONNX model
onnx_model = ort.InferenceSession(model_path)

# Custom transform function using PIL and NumPy
def transform_image(image):
    # Resize to 224x224
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    # Convert to RGB and then to NumPy array
    image = np.array(image, dtype=np.float32)
    # Ensure image is in RGB format (some images might be RGBA)
    if image.shape[-1] != 3:
        image = image[..., :3]
    # Normalize to [0, 1]
    image = image / 255.0
    # Change from HWC (height, width, channels) to CHW (channels, height, width)
    image = image.transpose(2, 0, 1)
    # Add batch dimension (1, C, H, W)
    image = np.expand_dims(image, axis=0)
    return image

def predict_skin_disease(image):
    image = image.convert("RGB")
    image = transform_image(image)

    ort_inputs = {onnx_model.get_inputs()[0].name: image}
    ort_outs = onnx_model.run(None, ort_inputs)
    
    logits = np.array(ort_outs[0])
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax
    pred = np.argmax(probs, axis=1)[0]  # Argmax
    confidence = probs[0, pred] * 100

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
        
        # Handle webcam image
        elif 'webcamImage' in request.files:
            file = request.files['webcamImage']
            if file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"webcam_{timestamp}.jpg"
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                image = Image.open(path)
                label, conf = predict_skin_disease(image)
                result = label
                confidence = conf
                img = filename
    
    return render_template('index.html', result=result, confidence=confidence, img=img)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=10000)