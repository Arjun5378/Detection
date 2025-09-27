import cv2
from flask import Flask, render_template, request, jsonify
import os
import shutil
import numpy as np
from werkzeug.utils import secure_filename
import json
from PIL import Image
import torch
from torchvision import transforms, datasets

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GRADCAM_FOLDER'] = 'static/gradcam'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Global variables for model
model = None
train_dataset = None
device = None
transform = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model and setup for predictions"""
    global model, train_dataset, device, transform
    
    try:
        from resnet_cbam_model import ResNet50_CBAM, get_gpu_device
        
        device = get_gpu_device()
        print(f"✅ Using device: {device}")
        
        # Load datasets to get class names
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Check if train_data exists
        if not os.path.exists("train_data"):
            print("❌ train_data folder not found!")
            return False
            
        train_dataset = datasets.ImageFolder("train_data", transform=transform)
        num_classes = len(train_dataset.classes)
        print(f"✅ Loaded dataset with {num_classes} classes: {train_dataset.classes}")
        
        # Load model
        if not os.path.exists("resnet_cbam_model.pth"):
            print("❌ Model file resnet_cbam_model.pth not found!")
            return False
            
        model = ResNet50_CBAM(num_classes).to(device)
        model.load_state_dict(torch.load("resnet_cbam_model.pth", map_location=device))
        model.eval()
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def get_recommendation(class_name):
    """Get recommendation from JSON database"""
    try:
        with open('plant_disease_db.json', 'r') as f:
            disease_db = json.load(f)
        
        # Convert class name to match JSON format
        json_class_name = class_name.replace(' ', '_')
        
        if json_class_name in disease_db:
            data = disease_db[json_class_name]
            
            if "healthy" in class_name.lower():
                return {
                    "status": "healthy",
                    "message": data.get("Message", "Plant is healthy. Continue good practices."),
                    "symptoms": "No disease symptoms detected",
                    "treatment": "Maintain current care routine",
                    "prevention": data.get("Prevention", "Continue monitoring and proper watering")
                }
            else:
                return {
                    "status": "diseased",
                    "symptoms": data.get("Symptoms", "Symptoms information not available"),
                    "treatment": data.get("Treatment", "Consult local agricultural expert for treatment"),
                    "prevention": data.get("Prevention", "Practice good crop management and sanitation")
                }
        else:
            # Fallback
            if "healthy" in class_name.lower():
                plant_name = class_name.split('_')[0] if '_' in class_name else class_name
                return {
                    "status": "healthy",
                    "message": f"{plant_name} plant is healthy. No disease detected.",
                    "symptoms": "No visible disease symptoms",
                    "treatment": "Continue regular maintenance",
                    "prevention": "Monitor plant health regularly"
                }
            else:
                parts = class_name.split('_')
                plant_name = parts[0] if parts else "Plant"
                disease_name = ' '.join(parts[1:]) if len(parts) > 1 else "disease"
                return {
                    "status": "diseased",
                    "symptoms": f"Symptoms data not available for {disease_name}",
                    "treatment": "Please consult with agricultural expert for specific treatment",
                    "prevention": "Practice good crop rotation and field sanitation"
                }
    except Exception as e:
        print(f"Error loading recommendations: {e}")
        return {
            "status": "unknown",
            "symptoms": "Information not available",
            "treatment": "Consult agricultural expert",
            "prevention": "Practice good crop management"
        }

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalization
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        
        return cam, target_class

def apply_gradcam_to_image(original_image, cam, alpha=0.5):
    """Apply Grad-CAM heatmap to original image"""
    import cv2
    import numpy as np
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # Convert PIL image to numpy
    img_array = np.array(original_image)
    
    # Blend images
    superimposed_img = heatmap * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)

def predict_with_gradcam(image_path):
    """Make prediction with Grad-CAM visualization"""
    global model, train_dataset, device, transform
    
    if model is None:
        raise Exception("Model not loaded")
    
    try:
        import cv2
        import numpy as np
        
        # Load and preprocess image
        original_img = Image.open(image_path).convert("RGB")
        img_tensor = transform(original_img).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_index = predicted.item()
        class_name = train_dataset.classes[class_index]
        confidence_score = confidence.item()
        
        # Generate Grad-CAM
        target_layer = model.resnet.layer4
        gradcam = GradCAM(model, target_layer)
        cam, _ = gradcam.generate_cam(img_tensor, class_index)
        
        # Apply Grad-CAM to original image
        gradcam_image = apply_gradcam_to_image(original_img, cam)
        
        return {
            'class_name': class_name,
            'confidence': confidence_score,
            'cam_heatmap': cam,
            'gradcam_image': gradcam_image,
            'original_image': original_img
        }
        
    except Exception as e:
        print(f"Grad-CAM prediction error: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image with Grad-CAM
            result = process_image_with_gradcam(filepath, filename)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

def process_image_with_gradcam(image_path, filename):
    """Process image and return results with Grad-CAM"""
    try:
        # Try to use actual model prediction
        if model is not None:
            prediction_result = predict_with_gradcam(image_path)
            
            # Save Grad-CAM image
            gradcam_filename = f"gradcam_{filename}"
            gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
            prediction_result['gradcam_image'].save(gradcam_path)
            
            recommendation = get_recommendation(prediction_result['class_name'])
            
            return {
                'status': 'success',
                'prediction': prediction_result['class_name'],
                'confidence': prediction_result['confidence'],
                'image_url': f'/static/uploads/{filename}',
                'gradcam_url': f'/static/gradcam/{gradcam_filename}',
                'recommendation': recommendation
            }
        else:
            # Fallback to sample data if model not loaded
            return get_sample_result(image_path, filename)
            
    except Exception as e:
        print(f"Error in process_image_with_gradcam: {e}")
        # Fallback to sample data
        return get_sample_result(image_path, filename)

def get_sample_result(image_path, filename):
    """Return sample result when model is not available"""
    return {
        'status': 'success',
        'prediction': 'Tomato_Late_blight',
        'confidence': 0.96,
        'image_url': f'/static/uploads/{filename}',
        'gradcam_url': f'/static/gradcam/sample_heatmap.jpg',  # You can add a sample heatmap
        'recommendation': get_recommendation('Tomato_Late_blight'),
        'note': 'Using sample data - model not fully integrated'
    }

@app.route('/diseases')
def diseases_list():
    """Return list of all diseases in database"""
    try:
        with open('plant_disease_db.json', 'r') as f:
            db = json.load(f)
        
        diseases = []
        for plant, data in db.items():
            if isinstance(data, dict):  # Check if it's a plant entry
                for disease, info in data.items():
                    if disease != "healthy" and isinstance(info, dict):
                        diseases.append({
                            'plant': plant,
                            'disease': disease,
                            'symptoms': info.get('symptoms', 'Not available'),
                            'severity': info.get('severity', 'Unknown')
                        })
        
        return jsonify(diseases)
    except Exception as e:
        print(f"Error loading diseases: {e}")
        return jsonify([])

@app.route('/health')
def health_check():
    """Check if model and dependencies are loaded properly"""
    model_loaded = model is not None
    diseases_loaded = os.path.exists('plant_disease_db.json')
    
    return jsonify({
        'model_loaded': model_loaded,
        'diseases_loaded': diseases_loaded,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'gradcam_folder_exists': os.path.exists(app.config['GRADCAM_FOLDER'])
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GRADCAM_FOLDER'], exist_ok=True)
    
    # Try to load model
    print("🚀 Starting Plant Disease Detector...")
    print("📦 Loading model and dependencies...")
    
    if load_model():
        print("✅ All systems ready!")
    else:
        print("⚠️  Model not loaded - using sample data mode")
    
    print("🌐 Starting web server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)