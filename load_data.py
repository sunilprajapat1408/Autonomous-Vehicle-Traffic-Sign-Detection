import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.utils import to_categorical
import shutil

def create_test_set():
    """Create test set from training data if it doesn't exist"""
    train_dir = os.path.join('Dataset', 'Train')
    test_dir = os.path.join('Dataset', 'Test')
    
    if not os.path.exists('Dataset'):
        raise FileNotFoundError("Dataset directory not found. Please run download_dataset.py first.")
        
    if not os.path.exists(train_dir):
        raise FileNotFoundError("Train directory not found. Please run download_dataset.py first.")
    
    if not os.path.exists(test_dir):
        print("Creating test set from training data (20% split)...")
        os.makedirs(test_dir, exist_ok=True)
        
        for class_folder in os.listdir(train_dir):
            train_path = os.path.join(train_dir, class_folder)
            test_path = os.path.join(test_dir, class_folder)
            
            if os.path.isdir(train_path):
                os.makedirs(test_path, exist_ok=True)
                images = [f for f in os.listdir(train_path) if f.endswith('.png')]
                num_test = int(len(images) * 0.2)
                test_images = np.random.choice(images, num_test, replace=False)
                
                for img_name in test_images:
                    src = os.path.join(train_path, img_name)
                    dst = os.path.join(test_path, img_name)
                    shutil.move(src, dst)

def preprocess_image(image_path):
    """Preprocess a single image"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 30x30
    image = cv2.resize(image, (30, 30))
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    return image

def load_data():
    """Load and preprocess the traffic sign dataset"""
    train_dir = os.path.join('Dataset', 'Train')
    test_dir = os.path.join('Dataset', 'Test')
    
    # Create test set if it doesn't exist
    create_test_set()
    
    X_train, y_train, X_test, y_test = [], [], [], []
    
    # Load training data
    print("Loading training data...")
    for class_id in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_id)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    try:
                        img_path = os.path.join(class_path, img_name)
                        img_array = preprocess_image(img_path)
                        X_train.append(img_array)
                        y_train.append(int(class_id))
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    # Load test data
    print("Loading test data...")
    for class_id in sorted(os.listdir(test_dir)):
        class_path = os.path.join(test_dir, class_id)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    try:
                        img_path = os.path.join(test_dir, class_id, img_name)
                        img_array = preprocess_image(img_path)
                        X_test.append(img_array)
                        y_test.append(int(class_id))
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=43)
    y_test = to_categorical(y_test, num_classes=43)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        load_data()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease follow these steps:")
        print("1. First run: python download_dataset.py")
        print("2. Then run: python load_data.py")