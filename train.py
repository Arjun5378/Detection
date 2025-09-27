import os
import shutil
import random

# Paths
DATASET_DIR = "PlantVillage"   # Original dataset folder
TRAIN_DIR = "train_data"
TEST_DIR = "test_data"

# Train-test split ratio
SPLIT_RATIO = 0.8  

def split_dataset():
    # Create train and test directories if not exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Loop through each class folder in dataset
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        # Compute split index
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Create class folders in train and test
        train_class_path = os.path.join(TRAIN_DIR, class_name)
        test_class_path = os.path.join(TEST_DIR, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        # Move files
        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_path, img))

        for img in test_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_path, img))

        print(f"✅ {class_name}: {len(train_imgs)} train, {len(test_imgs)} test")

    print("\n🎉 Dataset split complete!")
    print(f"Training data saved in: {TRAIN_DIR}")
    print(f"Testing data saved in: {TEST_DIR}")

if __name__ == "__main__":
    split_dataset()