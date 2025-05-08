# Deep Learning Face Recognition System

A comprehensive face recognition system using deep learning techniques with both PyQt5 and Tkinter GUI interfaces.

## Features

- **Advanced Face Detection**: Choose between MTCNN and enhanced YOLOv5 face detectors
- **Deep Learning Recognition**: Uses FaceNet embeddings and a deep neural network classifier
- **Multiple User Interfaces**: 
  - PyQt5-based GUI with tabbed interface for all face recognition tasks
  - Tkinter-based application with real-time controls and adjustable parameters
- **Real-Time Recognition**: Live webcam face detection and recognition
- **Face Registration**: Simple interface to add new faces to the database
- **Database Management**: View and manage registered faces
- **Configurable Settings**: Adjust model paths, camera settings, and confidence thresholds
- **Enhanced Unknown Face Detection**: Configurable thresholds for unknown face classification
- **Optimized Performance**: Enhanced YOLOv5 detector with improved hyperparameters and performance

## System Components

1. **Face Detection** (`face_detection.py`): Detects and extracts faces from images
2. **Face Recognition** (`face_recognition.py`): Generates embeddings and performs recognition
3. **Utilities** (`utils.py`): Helper functions for dataset handling and visualization
4. **Training** (`train_model.py`): Script for training the recognition model
5. **Real-Time Recognition** (`realtime_recognition.py`): Script for webcam recognition
6. **PyQt5 GUI Application** (`face_recognition_gui.py`): Main application with tabbed user interface
7. **Enhanced YOLOv5 Detector** (`enhanced_yolo_detector.py`): Improved YOLOv5 implementation
8. **Tkinter GUI Application** (`face_recognition_app.py`): Alternative GUI with real-time parameter adjustment

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download pre-trained models:
   - FaceNet model: [facenet_keras.h5](https://github.com/nyoki-mtl/keras-facenet/releases/download/v0.3.0/facenet_keras.h5)
   - YOLOv5 face model: [yolov5_face.pt](https://github.com/deepcam-cn/yolov5-face/releases)

   Note: The system will automatically fall back to using MTCNN if the YOLOv5 model is not found or if there are issues loading it.

## Usage

### GUI Applications

Run the PyQt5 GUI application to access all features in a tabbed interface:

```
python face_recognition_gui.py
```

Run the Tkinter GUI application for real-time parameter adjustment:

```
python face_recognition_app.py
```

### Training the Model

Train the recognition model with your dataset:

```
python train_model.py --dataset "Classified Dataset" --epochs 100 --augment
```

### Real-Time Recognition

Run real-time recognition from the command line:

```
# Using MTCNN detector
python realtime_recognition.py --detector mtcnn --display-fps

# Using  YOLOv5 detector (recommended for better performance)
python realtime_recognition.py --detector yolov5 --confidence 0.5 --display-fps
```

### Testing the Enhanced YOLOv5 Detector

Test the enhanced YOLOv5 face detector directly:

```
python enhanced_yolo_detector.py
```

## Dataset Structure

The face dataset should be organized as follows:

```
Classified Dataset/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

## Deep Learning Pipeline

1. **Face Detection**: Choose between two methods:
   - **MTCNN**: Traditional multi-task cascaded CNN for face detection
   - **Enhanced YOLOv5**: Optimized YOLOv5 model with improved hyperparameters for face detection
     - Lower default confidence threshold (0.5)
     - IoU threshold (0.45) for non-maximum suppression
     - Aspect ratio preservation during preprocessing
     - Increased maximum detections (50) for crowded scenes
2. **Face Embedding**: FaceNet generates 128-dimensional embeddings
3. **Face Classification**: Deep neural network classifies embeddings with configurable thresholds for unknown detection

## Requirements

- Python 3.6+
- TensorFlow 2.5+
- PyQt5 5.15+ (for PyQt5 GUI)
- Tkinter (for Tkinter GUI)
- OpenCV 4.5+
- MTCNN 0.1.0+
- PyTorch 1.8+ (for YOLOv5)
- Other dependencies in requirements.txt

## How to Implement

Follow these steps to implement the face recognition system from scratch:

1. **Set up the environment**:
   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv face_rec_env
   source face_rec_env/bin/activate  # On Windows: face_rec_env\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Download pre-trained models**:
   ```bash
   # Create a models directory
   mkdir -p models
   
   # Download FaceNet model
   wget -O models/facenet_keras.h5 https://github.com/nyoki-mtl/keras-facenet/releases/download/v0.3.0/facenet_keras.h5
   
   # Download YOLOv5 face model (optional, system will fall back to MTCNN if not available)
   wget -O models/yolov5_face.pt https://github.com/deepcam-cn/yolov5-face/releases/download/v0.3.0/yolov5n-face.pt
   ```

3. **Prepare your dataset**:
   ```bash
   # Create dataset structure
   mkdir -p "Classified Dataset"
   mkdir -p "Classified Dataset/person1"
   mkdir -p "Classified Dataset/person2"
   # Add images to each person's folder
   ```

4. **Implement core components** (in this order):
   - First implement `utils.py` with helper functions
   - Then implement `face_detection.py` with the FaceDetector class
   - Next implement `face_recognition.py` with the FaceRecognizer class
   - Implement `enhanced_yolo_detector.py` for the improved YOLOv5 detector

5. **Implement training and recognition scripts**:
   - Create `train_model.py` for model training
   - Create `realtime_recognition.py` for command-line recognition

6. **Implement GUI applications**:
   - Create `face_recognition_gui.py` for the PyQt5 interface
   - Create `face_recognition_app.py` for the Tkinter interface

7. **Train your model**:
   ```bash
   python train_model.py --dataset "Classified Dataset" --epochs 100 --augment
   ```

8. **Run the application**:
   # Run Tkinter GUI
   python face_recognition_app.py


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
            from enhanced_yolo_detector import SimpleYOLOv5Detector
            self.detector = SimpleYOLOv5Detector()
    
    def detect_faces(self, image, confidence_threshold=0.5):
        # Implementation for face detection
        # Returns list of face bounding boxes
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

#### GUI Implementation
```python
# In face_recognition_gui.py
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtCore import QThread, pyqtSignal
import cv2

class RecognitionThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    
    def __init__(self, face_detector, face_recognizer):
        super().__init__()
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        # ...
    
    def run(self):
        # Capture video and perform recognition
        # ...

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up UI with tabs
        # ...
```
