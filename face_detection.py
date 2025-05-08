import cv2
import numpy as np
from mtcnn import MTCNN
import torch
import os
from PIL import Image

# Import the YOLOv5 detector
try:
    from yolo_detector import YOLOv5FaceDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Simple YOLOv5 detector class for fallback when the full implementation is not available
class SimpleYOLOv5Detector:
    def __init__(self, model_path, device='cpu', conf_threshold=0.5):
        self.model = torch.load(model_path, map_location=device)
        self.device = device
        self.conf_threshold = conf_threshold
        
    def __call__(self, img):
        # Convert OpenCV BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 640x640 (standard YOLOv5 input size)
        img_resized = cv2.resize(img_rgb, (640, 640))
        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            pred = self.model(img_tensor)
        
        # Return predictions above threshold
        return pred[0][pred[0][:, 4] > self.conf_threshold]

class FaceDetector:
    def __init__(self, detector_type='mtcnn', confidence_threshold=0.8, device=None, verbose=True):
        self.detector_type = detector_type.lower()
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        # Set device for PyTorch models
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize the detector based on type
        if self.detector_type == 'mtcnn':
            self._init_mtcnn()
        elif self.detector_type == 'yolov5':
            self._init_yolov5()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}. "
                           f"Supported types are 'mtcnn' and 'yolov5'.")
    
    def _init_mtcnn(self):
        self.detector = MTCNN()
        if self.verbose:
            print("MTCNN detector initialized.")
    
    def _init_yolov5(self):
        # First try the root directory model file
        root_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5s.pt')
        # Fallback to models directory
        models_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yolov5s.pt')
        
        # Check if model file exists in root directory first
        if os.path.exists(root_model_path):
            model_path = root_model_path
            if self.verbose:
                print(f"Using YOLOv5s model from root directory: {model_path}")
        # Then check models directory
        elif os.path.exists(models_dir_path):
            model_path = models_dir_path
            if self.verbose:
                print(f"Using YOLOv5s model from models directory: {model_path}")
        else:
            if self.verbose:
                print(f"YOLOv5s model not found. Falling back to MTCNN.")
            self.detector_type = 'mtcnn'
            self._init_mtcnn()
            return
        
        try:
            if YOLO_AVAILABLE:
                if self.verbose:
                    print("Loading YOLOv5s detector...")
                self.detector = YOLOv5FaceDetector(
                    model_path=model_path,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                    enable_timing=True  # Enable performance monitoring
                )
                if self.verbose:
                    print("YOLOv5s detector loaded successfully!")
            else:
                if self.verbose:
                    print("YOLOv5s detector not available. Using basic implementation...")
                self.detector = SimpleYOLOv5Detector(
                    model_path, 
                    device=self.device, 
                    conf_threshold=self.confidence_threshold
                )
                if self.verbose:
                    print("Basic YOLOv5s model loaded successfully!")
        except Exception as e:
            if self.verbose:
                print(f"Failed to load YOLOv5s model: {e}\nFalling back to MTCNN.")
            self.detector_type = 'mtcnn'
            self._init_mtcnn()
    
    def detect_faces(self, image):
        """Detect faces in an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of dictionaries containing face information
        """
        try:
            if self.detector_type == 'mtcnn':
                return self._detect_faces_mtcnn(image)
            elif self.detector_type == 'yolov5':
                return self._detect_faces_yolov5(image)
            else:
                return []
        except Exception as e:
            if self.verbose:
                print(f"Error during face detection: {e}")
            return []
    
    def _detect_faces_mtcnn(self, image):
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)
        
        # Filter by confidence
        return [d for d in detections if d['confidence'] >= self.confidence_threshold]
    
    def _detect_faces_yolov5(self, image):
        if YOLO_AVAILABLE and isinstance(self.detector, YOLOv5FaceDetector):
            return self.detector.detect_faces(image)
        #Fallback
        orig_h, orig_w = image.shape[:2]
        detections = self.detector(image)
        
        # Process detections
        faces = []
        if detections is not None and len(detections) > 0:
            # Convert from normalized coordinates to pixel coordinates
            for detection in detections:
                # YOLOv5 format: [x1, y1, x2, y2, confidence, class]
                if len(detection) >= 5:
                    x1, y1, x2, y2, conf = detection[:5]
                    
                    # Scale from 640x640 to original image size
                    x1 = int(x1 * orig_w / 640)
                    y1 = int(y1 * orig_h / 640)
                    x2 = int(x2 * orig_w / 640)
                    y2 = int(y2 * orig_h / 640)
                    
                    # Create face info dictionary
                    faces.append({
                        'box': [x1, y1, x2-x1, y2-y1],
                        'confidence': float(conf),
                        'keypoints': {}  # YOLOv5 doesn't provide keypoints by default
                    })
        return faces
    
    def extract_face(self, image, face_info, required_size=(160, 160)):
        x, y, width, height = face_info['box']
        
        # Ensure coordinates are within image boundaries
        x, y = max(0, x), max(0, y)
        width = min(width, image.shape[1] - x)
        height = min(height, image.shape[0] - y)
        
        # Extract the face
        face = image[y:y+height, x:x+width]
        
        # Convert to PIL Image for resizing
        face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_image = face_image.resize(required_size)
        
        # Convert back to numpy array
        return cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    
    def detect_and_extract_faces(self, image, required_size=(160, 160)):
        # Detect faces
        faces_info = self.detect_faces(image)
        
        # Extract each face
        extracted_faces = []
        for face_info in faces_info:
            face_image = self.extract_face(image, face_info, required_size)
            extracted_faces.append((face_image, face_info))
            
        return extracted_faces
    
    def draw_faces(self, image, faces_info, draw_keypoints=True):
        img_with_faces = image.copy()
        
        for face_info in faces_info:
            # Extract face box coordinates
            x, y, width, height = face_info['box']
            
            # Draw rectangle around face
            cv2.rectangle(img_with_faces, (x, y), (x+width, y+height), (0, 255, 0), 2)
            
            # Add confidence text
            confidence = face_info['confidence']
            text = f"{confidence:.2f}"
            cv2.putText(img_with_faces, text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints if available and requested
            if draw_keypoints and 'keypoints' in face_info and face_info['keypoints']:
                for key, point in face_info['keypoints'].items():
                    cv2.circle(img_with_faces, point, 2, (0, 0, 255), 2)
        
        return img_with_faces
