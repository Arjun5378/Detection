# grad_cam.py
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into the target layer
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

def save_gradcam_visualization(original_image, cam, output_path, alpha=0.5):
    """Save Grad-CAM visualization"""
    result_image = apply_gradcam_to_image(original_image, cam, alpha)
    result_image.save(output_path)
    return output_path