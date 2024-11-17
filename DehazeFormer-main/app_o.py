from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import subprocess
import sys
import torch

app = Flask(__name__, static_folder='static')

print("Debug: ", sys.executable)

# Configure upload folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data/mu/test/hazy')
GT_FOLDER = os.path.join(BASE_DIR, 'data/mu/test/GT')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results/mu/dehazeformer-s/imgs')

# Ensure all required directories exist
for folder in [UPLOAD_FOLDER, GT_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Route for serving the frontend
@app.route('/')
def index():
    return send_file('static/index.html')

# Routes for serving images
@app.route('/data/mu/test/hazy/<path:filename>')
def serve_hazy(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/mu/dehazeformer-s/imgs/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Secure the filename
        filename = secure_filename(image.filename)
        
        # Save uploaded image
        hazy_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(hazy_path)
        
        # Create random GT image
        img = Image.open(hazy_path)
        black_img = Image.new("RGB", img.size, (0, 0, 0))  # Create an all-black image
        gt_path = os.path.join(GT_FOLDER, filename)
        black_img.save(gt_path)
        
        # Run DehazeFormer
        cmd = [
            sys.executable,
            "test.py",
            "--data_dir", "./data/",
            "--result_dir", "./results/",
            "--model", "dehazeformer-s",
            "--dataset", "mu",
            "--save_dir", "./saved_models/", 
            "--exp", "outdoor"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"DehazeFormer failed: {stderr.decode()}")
        
        return jsonify({
            'filename': filename,
            'message': 'Image processed successfully',
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'details': {
                'type': str(type(e).__name__),
                'args': str(e.args)
            }
        }), 500

if __name__ == '__main__':
    app.run(debug=True)