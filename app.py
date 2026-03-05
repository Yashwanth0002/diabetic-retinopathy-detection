# app.py
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import io
import base64

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Needed for flash messages

# Load your trained model
model = tf.keras.models.load_model('best_model.h5')

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Class names
class_names = [
    'No DR',
    'Mild',
    'Moderate', 
    'Severe',
    'Proliferative DR'
]

class_descriptions = {
    'No DR': 'No Diabetic Retinopathy',
    'Mild': 'Mild Nonproliferative Retinopathy',
    'Moderate': 'Moderate Nonproliferative Retinopathy',
    'Severe': 'Severe Nonproliferative Retinopathy', 
    'Proliferative DR': 'Proliferative Retinopathy'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Predict the class of an uploaded image"""
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = float(np.max(predictions[0]))
    
    # Get all probabilities
    all_probs = {class_names[i]: float(prob) for i, prob in enumerate(predictions[0])}
    
    # Create probability plot
    plt.figure(figsize=(10, 6))
    colors = ['green', 'lightblue', 'orange', 'red', 'darkred']
    bars = plt.bar(class_names, predictions[0], color=colors)
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, prob in zip(bars, predictions[0]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2%}', ha='center', va='bottom')
    
    # Save plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    plot_data = base64.b64encode(img_bytes.getvalue()).decode()
    plt.close()
    
    return {
        'predicted_class': predicted_class,
        'class_description': class_descriptions[predicted_class],
        'confidence': confidence,
        'all_predictions': all_probs,
        'plot_data': plot_data,
        'image_filename': os.path.basename(image_path)
    }

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/precautions')
def precautions():
    return render_template('precautions.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        try:
            result = predict_image(filepath)
            result['image_path'] = f'uploads/{filename}'
            return render_template('result.html', result=result)
        except Exception as e:
            flash(f'Prediction failed: {str(e)}')
            return redirect(url_for('home'))
    
    flash('Invalid file type. Please upload PNG, JPG, or JPEG.')
    return redirect(url_for('home'))

# API endpoint for AJAX requests (optional)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = predict_image(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)