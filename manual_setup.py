import os
import shutil

def manual_setup():
    # Path to the GTSRB directory
    gtsrb_dir = os.path.join('Dataset', 'GTSRB')
    
    if not os.path.exists(gtsrb_dir):
        print(f"Error: Could not find GTSRB directory at {gtsrb_dir}")
        return
    
    # Create Train directory
    train_dir = os.path.join('Dataset', 'Train')
    os.makedirs(train_dir, exist_ok=True)
    
    print("Setting up dataset structure...")
    try:
        # Look for the Final_Training/Images directory
        training_dir = os.path.join(gtsrb_dir, 'Final_Training', 'Images')
        
        if not os.path.exists(training_dir):
            print(f"Error: Could not find training images at {training_dir}")
            return
        
        # Copy each class folder
        for class_folder in os.listdir(training_dir):
            src = os.path.join(training_dir, class_folder)
            dst = os.path.join(train_dir, class_folder.zfill(2))  # Ensure 2-digit folder names
            
            if os.path.isdir(src):
                print(f"Moving class {class_folder}...")
                if os.path.exists(dst):
                    shutil.rmtree(dst)  # Remove if exists
                shutil.copytree(src, dst)
        
        print("Setup complete!")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        return

if __name__ == "__main__":
    manual_setup() 