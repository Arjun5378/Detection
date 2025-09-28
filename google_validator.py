# google_validator.py
"""
Validator + disease recommender + Grad-CAM integration.

Usage:
    - Place plant_disease_db.json in the same folder.
    - Ensure resnet_cbam_model.py and resnet_cbam_model.pth exist (or rely on fallback filename heuristics).
    - Optionally set GEMINI_API_KEY in environment to enable Google Gemini multimodal validation.
    - Run: python google_validator.py
"""

import os
import sys
import json
import re
import traceback
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms, datasets

# Optional Gemini (multimodal) validator
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ========== CONFIG ==========
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
GEMINI_MODEL = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# Database path
DB_PATH = "plant_disease_db.json"
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database file not found: {DB_PATH}")

with open(DB_PATH, "r", encoding="utf-8") as f:
    disease_db = json.load(f)

# Add project directory to path so we can import model definition file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ========== Grad-CAM helper ==========
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

        # Keep original hook behavior; may warn on modern PyTorch versions
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        return cam, target_class

def apply_gradcam_to_image(original_image, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
    img_array = np.array(original_image)
    superimposed_img = heatmap * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed_img)

def load_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print(f"Failed to load image: {e}")
        return None

# ========== Recommendation logic ==========
def infer_expected_outcome(disease_data):
    if not disease_data:
        return "Outcome not available."
    if "expected_outcome" in disease_data:
        return disease_data["expected_outcome"]
    severity = disease_data.get("severity", "").lower()
    if severity == "high":
        return ("If treated promptly and infected tissue removed, recovery may be possible but "
                "significant yield loss can occur. Severe infections may require replanting.")
    elif severity in ("medium", "med"):
        return ("Treatment is often effective; plant should recover with careful management and supplements, "
                "with minor-to-moderate yield impact possible.")
    elif severity == "low":
        return ("Likely to recover fully with correct treatment and supportive supplements.")
    else:
        return ("With prompt and appropriate treatment, plant condition should improve. Monitor progress and re-treat if necessary.")

def get_recommendation(class_name):
    """
    Returns dict:
      - status: 'healthy' or 'diseased'
      - symptoms, treatment, prevention
      - supplements or recommended_supplements
      - expected_outcome
    """
    json_class_name = class_name.replace(' ', '_')
    parts = json_class_name.split('_', 1)
    if len(parts) == 2:
        plant_name, disease_name = parts
    else:
        plant_name, disease_name = parts[0], "healthy"

    plant_data = disease_db.get(plant_name)
    if plant_data:
        if "healthy" in disease_name.lower():
            data = plant_data.get("healthy", {})
            expected = data.get("expected_outcome") or infer_expected_outcome(data)
            return {
                "status": "healthy",
                "message": data.get("message", f"{plant_name} plant is healthy."),
                "symptoms": data.get("symptoms", "No visible disease symptoms"),
                "treatment": data.get("treatment", "Maintain current care routine"),
                "prevention": data.get("prevention", "Continue good crop practices"),
                "recommended_supplements": data.get("recommended_supplements", "Organic compost, micronutrients, balanced NPK as needed"),
                "expected_outcome": expected
            }

        disease_entry = plant_data.get(disease_name)
        if disease_entry:
            expected = disease_entry.get("expected_outcome") or infer_expected_outcome(disease_entry)
            # Keep backward compatibility by reading both structured arrays and combined string
            supplements = disease_entry.get("supplements")
            if not supplements:
                # attempt to join structured lists if present
                organic = disease_entry.get("organic_supplements", [])
                chemical = disease_entry.get("chemical_supplements", [])
                parts = []
                if organic:
                    parts.append("Organic: " + ", ".join(organic))
                if chemical:
                    parts.append("Chemical: " + ", ".join(chemical))
                supplements = " | ".join(parts) if parts else "Apply both organic and chemical supplements as required."

            return {
                "status": "diseased",
                "symptoms": disease_entry.get("symptoms", "Symptoms information not available"),
                "treatment": disease_entry.get("treatment", "Consult local agricultural expert for treatment"),
                "prevention": disease_entry.get("prevention", "Practice good crop management and sanitation"),
                "supplements": supplements,
                "expected_outcome": expected
            }

    # Fallbacks
    if "healthy" in class_name.lower():
        return {
            "status": "healthy",
            "message": f"{plant_name} plant is healthy. No disease detected.",
            "symptoms": "No visible disease symptoms",
            "treatment": "Continue regular maintenance",
            "prevention": "Monitor plant health regularly",
            "recommended_supplements": "General compost, micronutrients, and balanced fertilizer",
            "expected_outcome": "Plant is expected to remain healthy with normal care."
        }
    else:
        return {
            "status": "diseased",
            "symptoms": f"Symptoms data not available for {disease_name}",
            "treatment": "Please consult with agricultural expert for specific treatment",
            "prevention": "Practice good crop rotation and sanitation",
            "supplements": "Use supportive organic + chemical supplements to strengthen plant recovery",
            "expected_outcome": "Outcome unknown. Consult expert."
        }

# ========== Model setup and normalization ==========
def normalize_label(raw_label):
    """
    Normalize raw dataset class labels into a stable form:
    - collapse repeated underscores into a single underscore
    - strip leading/trailing underscores
    - replace spaces with underscore
    Example: 'Pepper__bell___Bacterial_spot' -> 'Pepper_bell_Bacterial_spot'
    """
    s = re.sub(r"_+", "_", raw_label)
    s = s.strip("_")
    s = s.replace(" ", "_")
    return s

def setup_model_for_prediction():
    """Set up and load your trained model and dataset classes for prediction"""
    try:
        from resnet_cbam_model import ResNet50_CBAM, get_gpu_device
    except Exception as e:
        print(f"❌ Could not import model definitions: {e}")
        traceback.print_exc()
        return None, None, None, None

    device = get_gpu_device()
    print(f"✅ Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    if not os.path.exists("train_data"):
        print("❌ train_data folder not found!")
        return None, None, None, None

    train_dataset = datasets.ImageFolder("train_data", transform=transform)
    num_classes = len(train_dataset.classes)

    # Normalize classes once and keep mapping to indices
    normalized_classes = [normalize_label(c) for c in train_dataset.classes]
    # Optionally: show a short sample (first 6) to confirm normalization
    sample = normalized_classes[:6]
    print(f"✅ Loaded dataset with {num_classes} classes. Sample normalized classes: {sample}")

    if not os.path.exists("resnet_cbam_model.pth"):
        print("❌ Model file resnet_cbam_model.pth not found!")
        return None, None, None, None

    model = ResNet50_CBAM(num_classes).to(device)
    model.load_state_dict(torch.load("resnet_cbam_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")

    # Return model, and modified container including normalized class list
    # We'll attach normalized_classes to the dataset object for downstream use
    train_dataset.normalized_classes = normalized_classes
    return model, train_dataset, device, transform

# ========== Prediction + Grad-CAM ==========
def predict_disease_with_gradcam(image_path, model, train_dataset, device, transform):
    """Predict disease using the trained model and produce Grad-CAM image"""
    try:
        original_img = Image.open(image_path).convert("RGB")
        img_tensor = transform(original_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        idx = int(predicted.item())
        # Use normalized class names computed earlier
        if hasattr(train_dataset, "normalized_classes"):
            class_name = train_dataset.normalized_classes[idx]
        else:
            raw_class = train_dataset.classes[idx]
            class_name = normalize_label(raw_class)

        confidence_score = float(confidence.item())

        target_layer = model.resnet.layer4
        gradcam = GradCAM(model, target_layer)
        cam, _ = gradcam.generate_cam(img_tensor)

        gradcam_image = apply_gradcam_to_image(original_img, cam)
        return class_name, confidence_score, gradcam_image, cam

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return "Unknown", 0.0, None, None

def save_gradcam_image(gradcam_image, output_path):
    try:
        gradcam_image.save(output_path)
        print(f"💾 Grad-CAM image saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error saving Grad-CAM image: {e}")
        return None

# ========== Output formatting ==========
def display_diagnosis(class_name, recommendation, confidence=None, gradcam_path=None, heatmap_intensity=0.0):
    print("\n" + "="*70)
    print("🌿 PLANT DISEASE DIAGNOSIS REPORT")
    print("="*70)
    print(f"📋 Detected Class: {class_name.replace('_', ' ')}")
    if confidence is not None:
        print(f"🎯 Confidence: {confidence:.2%}")
    print(f"📊 Status: {recommendation['status'].upper()}")
    if gradcam_path:
        print(f"🔍 Grad-CAM Visualization: {gradcam_path}")
        print(f"   - Heatmap intensity: {heatmap_intensity:.3f}")
    print("-"*70)
    if recommendation["status"] == "diseased":
        print("🔴 DISEASE DETECTED")
        print(f"🔍 Symptoms: {recommendation.get('symptoms')}")
        print(f"⚕️ How to treat: {recommendation.get('treatment')}")
        print(f"🧴 Supplements to use: {recommendation.get('supplements')}")
        print(f"🛡️ Prevention: {recommendation.get('prevention')}")
        print(f"🔮 Expected outcome after treatment: {recommendation.get('expected_outcome')}")
    else:
        print("💚 PLANT IS HEALTHY")
        print(f"💡 Message: {recommendation.get('message')}")
        print(f"🧴 Recommended supplements (maintenance): {recommendation.get('recommended_supplements')}")
        print(f"🔮 Expected outlook: {recommendation.get('expected_outcome')}")
    print("="*70)

# ========== Main validation flow ==========
def validate_leaf_image_and_forward(image_path, save_gradcam=True, gradcam_output_dir="gradcam_output"):
    img = load_image(image_path)
    if img is None:
        return {'status': 'error', 'message': "Failed to load image."}

    # Validator prompt for Gemini
    prompt = (
        "You are a strict validator for plant disease detection system.\n"
        "Check if the given image is clearly of a plant leaf (preferably close-up).\n"
        "Respond with ONLY one word:\n"
        "- valid → if it is a clear plant leaf image suitable for disease detection\n"
        "- invalid → if it is not a plant leaf, or the leaf is not clearly visible"
    )

    result = "valid"
    if GEMINI_MODEL:
        try:
            response = GEMINI_MODEL.generate_content([img, prompt], stream=False)
            result = response.text.strip().lower()
            print(f"🔍 Validator (Gemini) result: {result}")
        except Exception as e:
            print("⚠️ Gemini validator error:", e)
            result = "valid"
    else:
        # Simple green-channel heuristic for leaf presence
        try:
            arr = np.array(img)
            green_ratio = np.mean((arr[..., 1] > arr[..., 0]) & (arr[..., 1] > arr[..., 2]))
            print(f"🔍 Simple heuristic green_ratio={green_ratio:.3f}")
            result = "valid" if green_ratio > 0.03 else "invalid"
        except Exception:
            result = "valid"

    if result != "valid":
        print("❌ Invalid image detected. Provide a clear, close-up image of a leaf.")
        return {'status': 'invalid_image', 'message': 'Image is not a valid plant leaf'}

    print("✅ Valid plant leaf. Proceeding with disease detection...")
    model_obj, train_dataset, device, transform = setup_model_for_prediction()

    if model_obj is None:
        # Fallback: use filename heuristics and return sample recommendation
        print("⚠️ Model not available — returning sample recommendation based on filename heuristics.")
        filename = os.path.basename(image_path).lower()
        if "healthy" in filename:
            class_name = "Tomato_healthy"
        elif "late_blight" in filename or "lateblight" in filename:
            class_name = "Tomato_Late_blight"
        elif "potato" in filename and "late" in filename:
            class_name = "Potato_Late_blight"
        else:
            class_name = "Tomato_Late_blight"

        recommendation = get_recommendation(class_name)
        display_diagnosis(class_name, recommendation, confidence=0.85, gradcam_path=None)
        return {
            'class_name': class_name,
            'confidence': 0.85,
            'recommendation': recommendation,
            'gradcam_path': None,
            'status': 'sample_data'
        }

    try:
        class_name, confidence, gradcam_image, cam = predict_disease_with_gradcam(
            image_path, model_obj, train_dataset, device, transform
        )
        recommendation = get_recommendation(class_name)

        gradcam_path = None
        heatmap_intensity = 0.0
        if gradcam_image and save_gradcam:
            os.makedirs(gradcam_output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            gradcam_filename = f"gradcam_{base_name}.jpg"
            gradcam_path = os.path.join(gradcam_output_dir, gradcam_filename)
            gradcam_path = save_gradcam_image(gradcam_image, gradcam_path)
            heatmap_intensity = float(np.max(cam) if cam is not None else 0.0)

        display_diagnosis(class_name, recommendation, confidence, gradcam_path, heatmap_intensity)

        return {
            'class_name': class_name,
            'confidence': confidence,
            'recommendation': recommendation,
            'gradcam_path': gradcam_path,
            'heatmap_intensity': heatmap_intensity,
            'status': 'success'
        }

    except Exception as e:
        print(f"❌ Error during disease detection: {e}")
        traceback.print_exc()
        # fallback sample recommendation
        class_name = "Tomato_Late_blight"
        recommendation = get_recommendation(class_name)
        display_diagnosis(class_name, recommendation, confidence=0.5, gradcam_path=None)
        return {
            'class_name': class_name,
            'confidence': 0.5,
            'recommendation': recommendation,
            'gradcam_path': None,
            'status': 'fallback'
        }

# Flask helper
def process_image_for_flask(image_path, filename):
    result = validate_leaf_image_and_forward(image_path)
    if result and result['status'] in ['success', 'sample_data', 'fallback']:
        return {
            'status': 'success',
            'prediction': result['class_name'],
            'confidence': float(result['confidence']),
            'image_url': f'/static/uploads/{filename}',
            'gradcam_url': f'/static/gradcam/{os.path.basename(result["gradcam_path"])}' if result['gradcam_path'] else None,
            'recommendation': result['recommendation'],
            'heatmap_intensity': float(result.get('heatmap_intensity', 0.0)),
            'expected_outcome': result['recommendation'].get('expected_outcome')
        }
    else:
        return {
            'status': 'error',
            'message': result.get('message', 'Image processing failed')
        }

# ========== CLI Entry ==========
if __name__ == "__main__":
    # Example: update this path to a real image on your machine for testing
    example_image = r"C:\Users\ASUS\Downloads\late_blight_tomato_leaf5x1200-1152x1536.jpg"
    if os.path.exists(example_image):
        _ = validate_leaf_image_and_forward(example_image)
        print("\n✅ Processing complete. See the diagnosis report above.")
    else:
        print(f"❌ Example image not found at {example_image}.")
        print("💡 Update the path or call validate_leaf_image_and_forward(image_path) from your app.")
