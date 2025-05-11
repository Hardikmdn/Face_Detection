import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import shutil
from tqdm import tqdm

def load_dataset(dataset_path, face_detector=None, required_size=(160, 160)):
    face_images = []
    labels = []
    
    # Walk through all subdirectories
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_dir):
            continue
        
        print(f"Loading images for {person_name}...")
        
        # Process each image in the person's directory
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # Skip if not an image file
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # If face detector is provided, detect and extract faces
            if face_detector is not None:
                faces = face_detector.detect_and_extract_faces(img, required_size)
                
                if not faces:
                    print(f"Warning: No face detected in {img_path}")
                    continue
                
                # Use the first detected face
                face_img = faces[0][0]
            else:
                # Just resize the image if no face detector
                face_img = cv2.resize(img, required_size)
            
            # Add to dataset
            face_images.append(face_img)
            labels.append(person_name)
    
    return np.array(face_images), np.array(labels)

def augment_dataset(dataset_path, output_path, samples_per_class=100, face_detector=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create data generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Process each person's directory
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_dir):
            continue
        
        # Create output directory for this person
        output_person_dir = os.path.join(output_path, person_name)
        os.makedirs(output_person_dir, exist_ok=True)
        
        print(f"Augmenting data for {person_name}...")
        
        # Get list of image files
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"Warning: No images found for {person_name}")
            continue
        
        # Calculate how many augmented images to create per original image
        num_orig_images = len(image_files)
        augmentations_per_image = max(1, samples_per_class // num_orig_images)
        
        # Process each image
        for img_name in image_files:
            img_path = os.path.join(person_dir, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # If face detector is provided, detect and extract faces
            if face_detector is not None:
                faces = face_detector.detect_and_extract_faces(img)
                
                if not faces:
                    print(f"Warning: No face detected in {img_path}")
                    continue
                
                # Use the first detected face
                face_img = faces[0][0]
            else:
                face_img = img
            
            # Convert to RGB for augmentation
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_rgb = np.expand_dims(face_img_rgb, axis=0)
            
            # Copy the original image
            output_img_path = os.path.join(output_person_dir, img_name)
            cv2.imwrite(output_img_path, face_img)
            
            # Generate augmented images
            i = 0
            for batch in datagen.flow(face_img_rgb, batch_size=1):
                # Convert back to BGR for saving
                aug_img = cv2.cvtColor(batch[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Save augmented image
                aug_name = f"{os.path.splitext(img_name)[0]}_aug_{i}{os.path.splitext(img_name)[1]}"
                aug_path = os.path.join(output_person_dir, aug_name)
                cv2.imwrite(aug_path, aug_img)
                
                i += 1
                if i >= augmentations_per_image:
                    break
    
    return output_path

def split_dataset(face_images, labels, test_size=0.2, val_size=0.1):
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        face_images: Array of face images
        labels: Array of corresponding labels
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        
    Returns:
        Dictionary with train, val, and test data
    """
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        face_images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Then split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

def plot_training_history(history):
    """
    Plot training history of a Keras model.
    
    Args:
        history: History object returned by model.fit()
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    return fig

def create_model_directories():
    """
    Create necessary directories for model storage.
    
    Returns:
        Dictionary with paths to model directories
    """
    # Define base directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    
    return {
        'base_dir': base_dir,
        'classifier_model': os.path.join(base_dir, 'face_classifier_model.h5'),
        'yolov5_model': os.path.join(base_dir, 'yolov5_face.pt')
    }

def preprocess_image_for_display(image):
    """
    Preprocess an image for display in the GUI.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        Preprocessed image suitable for display
    """
    # Make a copy to avoid modifying the original
    img = image.copy()
    
    # Ensure image is in BGR format (OpenCV default)
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    # Resize if too large
    max_height = 600
    max_width = 800
    
    height, width = img.shape[:2]
    
    if height > max_height or width > max_width:
        # Calculate new dimensions
        if height > width:
            new_height = max_height
            new_width = int(width * (max_height / height))
        else:
            new_width = max_width
            new_height = int(height * (max_width / width))
        
        # Resize
        img = cv2.resize(img, (new_width, new_height))
    
    return img
