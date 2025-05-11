## Project Overview

This project implements a face recognition system that combines multiple face detection methods using deep learning techniques,

I developed this system to explore the practical applications of deep learning in computer vision, specifically focusing on face recognition tasks. The implementation demonstrates how different components (detection, embedding generation, and classification) combine to create a real-time face recognition model.

## What I Implemented

### Core Components

1. **Dual Face Detection Systems**
   - **MTCNN Detector**: Implemented a Multi-task Cascaded Convolutional Neural Network detector
   - **YOLOv5 Detector**: Integrated and optimized a YOLOv5-based face detector for improved speed and accuracy

2. **Deep Learning Recognition Pipeline**
   - **Feature Extraction**: Used InceptionV3 as a base model for generating face embeddings
   - **Custom Classifier**: Designed a deep neural network with residual connections for face classification
   - **Unknown Face Detection**: Implemented confidence thresholding for identifying unknown individuals

3. **User Interfaces**
   - **Tkinter GUI**: Created a lightweight interface with real-time parameter adjustment capabilities

4. **Utilities and Tools**
   - **Dataset Management**: Tools for dataset loading, augmentation, and preprocessing
   - **Performance Visualization**: Implemented visualization tools to compare detector performance
   - **Model Training**: Created a training pipeline with data augmentation and validation

## Why I Chose These Components

### Face Detection Methods

I implemented both MTCNN and YOLOv5 detectors to provide flexibility and performance options:

- **MTCNN**: Chosen for its accuracy and built-in facial landmark detection capabilities. While slower, it provides excellent precision for applications where speed is not critical.

- **YOLOv5**: Implemented for its superior speed (approximately 4.8x faster than MTCNN) and better performance in challenging conditions like poor lighting and partial occlusions. This makes it ideal for real-time applications.

The performance comparison between these detectors is visualized in the `detector_performance_visualization.py` module, demonstrating the trade-offs between speed and accuracy.

### Recognition Architecture

I chose InceptionV3 as the base embedding model due to its balance of accuracy and computational efficiency. The custom classifier uses residual connections to improve gradient flow during training and prevent the vanishing gradient problem, which is particularly important for deep networks.

Built a custom classifier fully connected head. The classifier architecture includes:
- Residual blocks with batch normalization and leaky ReLU activations
- Dropout layers for regularization to prevent overfitting
- L2 regularization on weights to improve generalization

### User Interface Design
- **Tkinter**: Offers a simpler, more lightweight interface with real-time parameter adjustment. This is perfect for testing and fine-tuning detection and recognition parameters on the fly.

## How I Implemented It

### Face Detection Implementation

The face detection module (`face_detection.py`) provides a unified interface for both MTCNN and YOLOv5 detectors:

```python
class FaceDetector:
    def __init__(self, detector_type='mtcnn', confidence_threshold=0.7):
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        
        if detector_type == 'mtcnn':
            self.detector = MTCNN(min_face_size=20, thresholds=[0.6, 0.7, 0.7])
        elif detector_type == 'yolov5':
            # Import YOLOv5 detector with optimized parameters
            try:
                from _yolo_detector import SimpleYOLOv5Detector
                self.detector = SimpleYOLOv5Detector(
                    confidence_threshold=confidence_threshold,
                    iou_threshold=0.45,
                    max_detections=50
                )
            except Exception as e:
                print(f"Error loading YOLOv5 detector: {str(e)}")
                print("Falling back to MTCNN detector")
                self.detector_type = 'mtcnn'
                self.detector = MTCNN(min_face_size=20, thresholds=[0.6, 0.7, 0.7])
```

The YOLOv5 detector was optimized with custom hyperparameters:
- Lower default confidence threshold (0.5) to detect more potential faces
- IoU threshold of 0.45 for non-maximum suppression to balance between duplicate detection and missing faces
- Aspect ratio preservation during preprocessing to maintain face proportions
- Increased maximum detections (50) to handle crowded scenes

### Face Recognition Implementation

The face recognition module (`face_recognition.py`) uses a multi-stage approach:

1. **Preprocessing**: Faces are resized to 160x160 pixels and normalized
2. **Embedding Generation**: The InceptionV3 model extracts 2048-dimensional feature vectors
3. **Classification**: A custom deep neural network with residual connections classifies the embeddings

The classifier architecture includes residual blocks to improve training:

```python
def residual_block(self, x, units, dropout_rate=0.3, l2_reg=0.0005):
    # Store the input for the residual connection
    residual = x
    
    # First dense layer
    x = Dense(units, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    # Second dense layer
    x = Dense(units, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    
    # If input and output dimensions don't match, use a projection shortcut
    if residual.shape[-1] != units:
        residual = Dense(units, kernel_regularizer=l2(l2_reg))(residual)
        residual = BatchNormalization()(residual)
    
    # Add the residual connection
    x = Add()([x, residual])
    x = LeakyReLU(alpha=0.1)(x)
    
    return x
```

### Dataset Management and Augmentation

The utilities module (`utils.py`) provides functions for dataset loading and augmentation:

```python
def augment_dataset(dataset_path, output_path, samples_per_class=100, face_detector=None):
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
        # Calculate augmentations per image
        num_orig_images = len(image_files)
        augmentations_per_image = max(1, samples_per_class // num_orig_images)
        
        # Generate augmented images
        for batch in datagen.flow(face_img_rgb, batch_size=1):
            aug_img = cv2.cvtColor(batch[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(aug_path, aug_img)
```

Data augmentation is crucial for improving model generalization, especially with limited training data. The augmentation pipeline applies random transformations (rotation, shift, zoom, flip) to increase dataset diversity.

## How to Use This System

### Required Directory Structure

Before starting, ensure you have the necessary directories:

```
Project/
├── models/                      # Contains model files (excluded from git)
│   ├── yolov5s.pt               # YOLOv5 model
│   ├── face_classifier_model.h5 # Trained classifier
├── Face Dataset/          # Original dataset (excluded from git)
│   ├── Unknown/                 # Unknown faces
│   ├── person1/                 # Person 1
│   ├── person2/                 # Person 2
│   ├── ...
├── Augmented Dataset/           # Augmented dataset (excluded from git)
│   ├── Unknown/                 # Unknown faces
│   ├── person1/                 # Person 1
│   ├── person2/                 # Person 2
│   ├── ...
└── [python files]               # Implementation files
```

> **Note:** The models and dataset directories are excluded from git tracking via .gitignore. You'll need to create these directories and download/add the required files manually.

### Step 1: Environment Setup

```bash
# Create and activate a virtual environment
python -m venv face_rec_env
source face_rec_env/bin/activate  # On Windows: face_rec_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Pre-trained Models

```bash
# Create models directory
mkdir -p models

# Download YOLOv5 face model
wget -O models/yolov5s.pt https://huggingface.co/Ultralytics/YOLOv5/blob/main/yolov5s.pt
```

### Step 3: Prepare Your Dataset

Create a dataset with the following structure:

```
Face Dataset/
├── Unknown/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Each person should have their own directory containing multiple face images.

It is recommended to have an Unknown File for preventing overfitting to classes. Personally, For this project, I pulled 500 random images from the Labelled Faces in the Wild (LFW) dataset.

### Step 4: Data Augmentation (Optional but Recommended)

```bash
python -c "from utils import augment_dataset; augment_dataset('Face Dataset', 'augmented_dataset', samples_per_class=100)"
```

This will create an augmented dataset with approximately 100 images per person, which helps improve model performance.

### Step 5: Train the Recognition Model
You are recommended to play witht the parameters to get the best results. 
```bash
python train_model.py --dataset "Face Dataset" --epochs 100  --augment --use-augmented  --detector mtcnn  --confidence 0.9
```

The trained model will be saved to `models/face_classifier_model.h5`.

### Step 6: Run the Application

Choose one of the GUI applications:

```bash
# Run the Tkinter GUI (real-time parameter adjustment)
python face_recognition_app.py

### Step 7: Using the Application

#### With the Tkinter GUI:

1. Adjust parameters in real-time:
   - Select detector type (MTCNN or YOLOv5)
   - Adjust detection confidence threshold
   - Set recognition threshold for unknown face detection
   - Toggle display options (FPS, confidence scores)

2. Click "Start Recognition" to begin real-time recognition

## Performance Considerations

- **YOLOv5 vs MTCNN**: YOLOv5 is approximately 4.8x faster than MTCNN but may have slightly lower precision in some cases
- **Batch Size**: Larger batch sizes during training can improve performance, but require more memory
- **Model Size**: The current implementation uses InceptionV3 for embeddings, which provides a good balance between accuracy and speed
- **Real-time Performance**: For optimal real-time performance, use the YOLOv5 detector with a confidence threshold of 0.5-0.7

## Troubleshooting

- **Model Loading Errors**: Ensure all model files are in the correct location (models directory)
- **CUDA/GPU Issues**: If using GPU acceleration, verify compatible CUDA and cuDNN versions for TensorFlow/PyTorch
- **Memory Errors**: Reduce batch size during training if encountering memory issues
- **Detection Issues**: Adjust confidence thresholds based on lighting conditions and camera quality


### Implementation Details

#### Face Detection Implementation
```python
# In face_detection.py
from mtcnn import MTCNN
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, detector_type='mtcnn'):
        self.detector_type = detector_type
        if detector_type == 'mtcnn':
            self.detector = MTCNN()
        elif detector_type == 'yolov5':
            # Import YOLOv5 detector
            from _yolo_detector import SimpleYOLOv5Detector
            self.detector = SimpleYOLOv5Detector()
    
    def detect_faces(self, image, confidence_threshold=0.5):
        # Implementation for face detection
        # Returns a list of face bounding boxes
        # ...
```

#### Face Recognition Implementation
```python
# In face_recognition.py
from tensorflow.keras.models import load_model
import numpy as np

class FaceRecognizer:
    def __init__(self, facenet_model_path, classifier_path=None):
        # Load FaceNet model
        self.facenet_model = load_model(facenet_model_path)
        
        # Load classifier if available
        if classifier_path and os.path.exists(classifier_path):
            self.classifier = load_model(classifier_path)
            # Load class names
            # ...
    
    def generate_embedding(self, face_image):
        # Preprocess face and generate embedding
        # ...
    
    def recognize_face(self, embedding, recognition_threshold=0.5):
        # Classify face embedding
        # Return name and confidence
        # ...
```
