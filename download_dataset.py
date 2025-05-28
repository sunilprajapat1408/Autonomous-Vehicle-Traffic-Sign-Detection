import os
import zipfile
import shutil
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import time

def download_dataset():
    try:
        print("Downloading dataset from Kaggle...")
        
        # Check if kaggle.json exists, if not, try to create it from user input
        kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_json):
            print("Kaggle API credentials not found. Please enter your Kaggle credentials:")
            username = input("Kaggle username: ")
            key = input("Kaggle API key: ")
            
            # Create .kaggle directory if it doesn't exist
            os.makedirs(kaggle_dir, exist_ok=True)
            
            # Save credentials
            credentials = {"username": username, "key": key}
            with open(kaggle_json, 'w') as f:
                json.dump(credentials, f)
            
            # Set permissions on Windows
            if os.name == 'nt':  # Windows
                import subprocess
                subprocess.run(['icacls', kaggle_json, '/inheritance:r', '/grant:r', f'{os.getenv("USERNAME")}:(R)'])
        
        # Initialize the Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Clean up any existing files
        if os.path.exists('Dataset'):
            shutil.rmtree('Dataset')
        if os.path.exists('temp_extract'):
            shutil.rmtree('temp_extract')
        if os.path.exists('gtsrb-german-traffic-sign.zip'):
            os.remove('gtsrb-german-traffic-sign.zip')
            
        os.makedirs('Dataset', exist_ok=True)
        
        # Download both train and test datasets
        print("Downloading GTSRB dataset... This may take a few minutes.")
        
        # Download training dataset
        api.dataset_download_files(
            'meowmeowmeowmeowmeow/gtsrb-german-traffic-sign',
            path='.',
            quiet=False
        )
        
        print("\nExtracting files...")
        with zipfile.ZipFile('gtsrb-german-traffic-sign.zip', 'r') as zip_ref:
            zip_ref.extractall('temp_extract')
        
        # Create Train and Test directories
        os.makedirs('Dataset/Train', exist_ok=True)
        os.makedirs('Dataset/Test', exist_ok=True)
        
        # Move training data
        source_train = 'temp_extract/Train'
        if os.path.exists(source_train):
            print("Processing training data...")
            for item in os.listdir(source_train):
                src = os.path.join(source_train, item)
                dst = os.path.join('Dataset/Train', item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    print(f"Processed training class {item}")
        
        # Create test set from training data (20% split)
        print("\nCreating test set from training data (20% split)...")
        import random
        from pathlib import Path
        
        for class_folder in os.listdir('Dataset/Train'):
            train_class_path = Path('Dataset/Train') / class_folder
            test_class_path = Path('Dataset/Test') / class_folder
            
            if train_class_path.is_dir():
                # Create test directory for this class
                test_class_path.mkdir(exist_ok=True)
                
                # Get all images in this class
                images = list(train_class_path.glob('*.png'))
                
                # Calculate number of test images (20% of total)
                num_test = int(len(images) * 0.2)
                
                # Randomly select test images
                test_images = random.sample(images, num_test)
                
                # Move test images to test directory
                for img_path in test_images:
                    shutil.move(str(img_path), str(test_class_path / img_path.name))
                
                print(f"Split class {class_folder}: {len(images)-num_test} train, {num_test} test")
        
        # Clean up
        print("\nCleaning up temporary files...")
        if os.path.exists('temp_extract'):
            shutil.rmtree('temp_extract')
        if os.path.exists('gtsrb-german-traffic-sign.zip'):
            os.remove('gtsrb-german-traffic-sign.zip')
        
        print("\nDataset downloaded and split successfully!")
        print("Train and Test datasets are now in the 'Dataset' folder")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nTo get your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to API section")
        print("3. Click 'Create New API Token'")
        print("4. This will download 'kaggle.json'")
        print("5. When prompted by this script, enter the username and key from that file")

if __name__ == "__main__":
    download_dataset() 