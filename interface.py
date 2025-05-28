import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from load_data import preprocess_image

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    pass

# Dictionary of traffic sign classes
classes = {
    0:'Speed limit (20km/h)', 
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow', 
    31:'Wild animals crossing',
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons'
}

def augment_and_predict(image, model, num_augmentations=5):
    """Perform multiple predictions with slight augmentations"""
    predictions = []
    
    # Original image prediction
    predictions.append(model.predict(np.expand_dims(image, axis=0), verbose=0)[0])
    
    # Slight rotations
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((15, 15), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (30, 30))
        predictions.append(model.predict(np.expand_dims(rotated, axis=0), verbose=0)[0])
    
    # Slight brightness variations
    for factor in [0.9, 1.1]:
        adjusted = np.clip(image * factor, 0, 1)
        predictions.append(model.predict(np.expand_dims(adjusted, axis=0), verbose=0)[0])
    
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

# Load the trained model
try:
    model = load_model('model_RTSR.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def classify(file_path):
    """Classify a traffic sign image"""
    try:
        # Read and preprocess the image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Could not read image")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image using the same function as training
        processed_image = preprocess_image(file_path)
        
        # Get prediction with augmentations
        predictions = augment_and_predict(processed_image, model)
        pred_class = np.argmax(predictions)
        confidence = predictions[pred_class]
        
        # Get sign name and update UI
        sign = classes[pred_class]
        update_ui_with_prediction(sign, confidence)
        print(f"Predicted class: {sign} (Confidence: {confidence:.2%})")
        
    except Exception as e:
        print(f"Error during classification: {e}")
        label.configure(text=f"Error: {str(e)}")
        confidence_label.configure(text="")

def update_ui_with_prediction(sign_name, confidence):
    """Update UI with prediction results"""
    label.configure(text=sign_name)
    confidence_label.configure(text=f"Confidence: {confidence:.2%}")

def show_classify_button(file_path):
    """Show classification button"""
    classify_btn = create_button(button_frame, "Classify Image", lambda: classify(file_path))

def upload_image():
    """Handle image upload"""
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:  # User cancelled
            return
            
        uploaded = Image.open(file_path)
        
        # Calculate size to maintain aspect ratio
        display_size = (400, 400)
        uploaded.thumbnail(display_size)
        
        im = ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text="Image loaded successfully")
        confidence_label.configure(text="Click 'Classify Image' to analyze")
        show_classify_button(file_path)
        
    except Exception as e:
        print(f"Error uploading image: {e}")
        label.configure(text=f"Error uploading image: {str(e)}")
        confidence_label.configure(text="")

# Initialize GUI
top = tk.Tk()
top.geometry('1000x800')
top.title('Road Traffic Sign Recognition')

# Create a modern dark theme
bg_color = '#1E1E1E'  # Dark background
accent_color = '#007ACC'  # Blue accent
text_color = '#FFFFFF'  # White text
button_color = '#404040'  # Gray buttons
success_color = '#4EC9B0'  # Teal for success messages

top.configure(background=bg_color)

# Create main container frame
main_frame = Frame(top, bg=bg_color)
main_frame.pack(expand=True, fill='both', padx=20, pady=20)

# Create and style the heading
heading_frame = Frame(main_frame, bg=bg_color)
heading_frame.pack(fill='x', pady=(0, 30))

heading = Label(heading_frame, 
               text="Road Traffic Sign Recognition",
               font=('Helvetica', 36, 'bold'),
               bg=bg_color,
               fg=accent_color)
heading.pack()

subheading = Label(heading_frame,
                  text="Upload a traffic sign image for instant recognition",
                  font=('Helvetica', 14),
                  bg=bg_color,
                  fg=text_color)
subheading.pack(pady=(5, 0))

# Create content frame
content_frame = Frame(main_frame, bg=bg_color)
content_frame.pack(expand=True, fill='both')

# Left frame for image
left_frame = Frame(content_frame, bg=bg_color)
left_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))

# Image container with border
image_container = Frame(left_frame, 
                       bg=accent_color,
                       padx=2, 
                       pady=2)
image_container.pack(pady=10)

sign_image = Label(image_container, bg=bg_color)
sign_image.pack(padx=10, pady=10)

# Right frame for results
right_frame = Frame(content_frame, bg=bg_color)
right_frame.pack(side='right', expand=True, fill='both', padx=(10, 0))

# Create a frame for the result
result_frame = Frame(right_frame, bg=bg_color)
result_frame.pack(fill='x', pady=10)

# Result label with modern styling
label = Label(result_frame,
             text="No image selected",
             font=('Helvetica', 18, 'bold'),
             bg=bg_color,
             fg=text_color,
             justify='left',
             wraplength=400)
label.pack(anchor='w')

# Confidence label
confidence_label = Label(result_frame,
                        text="",
                        font=('Helvetica', 14),
                        bg=bg_color,
                        fg=success_color)
confidence_label.pack(anchor='w', pady=(5, 0))

# Button frame
button_frame = Frame(main_frame, bg=bg_color)
button_frame.pack(fill='x', pady=20)

# Style the buttons
def create_button(parent, text, command):
    btn = Button(parent, 
                text=text,
                command=command,
                font=('Helvetica', 12, 'bold'),
                bg=button_color,
                fg=text_color,
                activebackground=accent_color,
                activeforeground=text_color,
                relief='flat',
                padx=20,
                pady=10,
                cursor='hand2')
    btn.pack(side='left', padx=5)
    return btn

# Create buttons
upload_btn = create_button(button_frame, "Select Image", upload_image)

# Start GUI
top.mainloop()
