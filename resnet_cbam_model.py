import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from nirmal_optimizer import NirmalOptimizer
from PIL import Image
import time
from datetime import datetime
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional

# Configuration
class TrainingConfig:
    def __init__(self):
        self.train_dir = "train_data"
        self.test_dir = "test_data"
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.patience = 7  # for early stopping
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
        self.model_save_path = "resnet_cbam_model.pth"
        self.best_model_save_path = "best_resnet_cbam_model.pth"

# Complete CBAM and ResNet50_CBAM implementation
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        avg_pool = nn.AdaptiveAvgPool2d(1)(x)
        channel_attention = self.channel_attention(avg_pool)
        x_channel = x * channel_attention
        
        # Spatial attention
        avg_pool_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)
        spatial_attention = self.spatial_attention(spatial_input)
        x_spatial = x_channel * spatial_attention
        
        return x_spatial

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_CBAM, self).__init__()
        self.num_classes = num_classes
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze early layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Add CBAM after layer4
        self.cbam = CBAMBlock(channels=2048)
        
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        # Unfreeze the final layers for training
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Standard ResNet forward pass until layer4
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Apply CBAM attention
        x = self.cbam(x)

        # Continue with ResNet forward pass
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x

class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = NirmalOptimizer(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.train_losses: List[float] = []
        self.val_accuracies: List[float] = []
        
        # Setup logging and directories
        self._setup_directories()
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary directories for checkpoints and logs"""
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.config.log_dir).mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
        
        # Save best model
        if is_best:
            torch.save(self.model.state_dict(), self.config.best_model_save_path)
            torch.save(checkpoint, f"{self.config.checkpoint_dir}/best_checkpoint.pth")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        return checkpoint['epoch']
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {self.current_epoch+1}/{self.config.epochs} | "
                    f"Batch {batch_idx}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}"
                )
        
        return running_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = val_loss / len(val_loader)
        return accuracy, avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              resume_checkpoint: Optional[str] = None):
        """Main training loop"""
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            start_epoch = self.load_checkpoint(resume_checkpoint)
            self.logger.info(f"Resumed training from epoch {start_epoch}")
        
        # Early stopping variables
        best_epoch = 0
        epochs_without_improvement = 0
        
        self.logger.info("Starting training...")
        self.logger.info(f"Training on {len(train_loader.dataset)} samples")
        self.logger.info(f"Validating on {len(val_loader.dataset)} samples")
        
        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_accuracy, val_loss = self.validate(val_loader)
            self.val_accuracies.append(val_accuracy)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} completed in {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.2f}% | LR: {current_lr:.6f}"
            )
            
            # Save checkpoint if best model
            is_best = val_accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = val_accuracy
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                self.save_checkpoint(epoch + 1, is_best=True)
                self.logger.info(f"New best model saved with accuracy: {val_accuracy:.2f}%")
            else:
                epochs_without_improvement += 1
                self.save_checkpoint(epoch + 1)
            
            # Early stopping
            if epochs_without_improvement >= self.config.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info(f"Training completed. Best accuracy: {self.best_accuracy:.2f}% at epoch {best_epoch}")

class PlantDiseasePredictor:
    def __init__(self, model: nn.Module, train_dataset: datasets.ImageFolder, 
                 device: torch.device, disease_db: dict):
        self.model = model
        self.train_dataset = train_dataset
        self.device = device
        self.disease_db = disease_db
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str) -> Dict:
        """Predict plant disease from image"""
        self.model.eval()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            class_name = self.train_dataset.classes[predicted.item()]
            confidence_score = confidence.item()
            
            # Parse result
            if "healthy" in class_name.lower():
                plant_name = class_name.split('_')[0]
                result = {
                    "status": "healthy",
                    "plant": plant_name,
                    "confidence": confidence_score,
                    "message": f"✅ Plant is healthy: {plant_name}",
                    "recommendation": self.disease_db.get(plant_name, {}).get("healthy", "No specific recommendations available.")
                }
            else:
                parts = class_name.split('_')
                plant_name = parts[0]
                disease_name = '_'.join(parts[1:]) if len(parts) > 1 else "Unknown"
                result = {
                    "status": "diseased",
                    "plant": plant_name,
                    "disease": disease_name,
                    "confidence": confidence_score,
                    "message": f"📋 Diagnosed: {plant_name} - {disease_name}",
                    "recommendation": self.disease_db.get(plant_name, {}).get(disease_name, "Consult with a plant expert for specific treatment.")
                }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing image: {str(e)}"
            }

def get_gpu_device():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available! Please run on a machine with CUDA-enabled GPU.")
    return torch.device("cuda")

def main():
    # Initialize configuration
    config = TrainingConfig()
    device = get_gpu_device()
    print(f"Using device: {device}")
    
    # Load disease database
    with open("plant_disease_db.json", "r") as f:
        disease_db = json.load(f)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(config.train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(config.test_dir, transform=val_transform)
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, 
                          shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=2)
    
    # Initialize model
    num_classes = len(train_dataset.classes)
    model = ResNet50_CBAM(num_classes).to(device)
    
    # Print model summary
    print(f"Model initialized with {num_classes} classes")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer and start training
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    test_accuracy, _ = trainer.validate(test_loader)
    trainer.logger.info(f"🎯 Final Test Accuracy: {test_accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), config.model_save_path)
    
    # Initialize predictor
    predictor = PlantDiseasePredictor(model, train_dataset, device, disease_db)
    
    # Example prediction (uncomment to test)
    # if os.path.exists("test_image.jpg"):
    #     result = predictor.predict("test_image.jpg")
    #     print(result["message"])
    #     print(result["recommendation"])

if __name__ == "__main__":
    main()