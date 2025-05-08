import cv2
import torch
import numpy as np
import os
import time
from PIL import Image

class YOLOv5FaceDetector:
    def __init__(
        self, 
        model_path=None,
        confidence_threshold=0.8,
        iou_threshold=0.45,        # NMS IoU threshold
        device=None,
        input_size=640,            # Input image size
        enable_timing=False        # Performance monitoring
    ):

        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'models', 'yolov5s.pt')
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Set parameters
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.enable_timing = enable_timing
        
        # Load model
        print(f"Loading YOLOv5s detector from local file (conf={confidence_threshold}, iou={iou_threshold})...")
        try:
            # Use direct loading from the local file
            print(f"Loading model from: {model_path}")
            self.model = torch.load(model_path, map_location=self.device)
            
            # Extract model from checkpoint if needed
            if isinstance(self.model, dict):
                if 'model' in self.model:
                    self.model = self.model['model']
                elif 'state_dict' in self.model:
                    self.model = self.model['state_dict']
            
            self.model.to(self.device)
            self.model.eval()
            self.use_torch_hub = False
            print(f"Successfully loaded YOLOv5s model from {model_path}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5s model: {e}")
            
        print(f"YOLOv5s detector initialized on {self.device}")
    
    def preprocess_image(self, image):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize with aspect ratio preservation
        h, w = img_rgb.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # Create canvas with padding
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = img_resized
        
        # Convert to tensor
        img_tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor, (scale, (w, h))
    
    def postprocess_detections(self, detections, image_info):
        scale, (orig_w, orig_h) = image_info
        faces = []
        
        # Process direct PyTorch model output
        if detections is not None and len(detections) > 0:
            # Handle different output formats
            if hasattr(detections, 'xyxy') and hasattr(detections, 'xywh'):
                # Handle YOLOv5 model output format
                for det in detections.xyxy[0]:
                    if len(det) >= 5 and float(det[4]) >= self.confidence_threshold:
                        x1, y1, x2, y2, conf = det[:5]
                        # Convert to original image coordinates
                        x1, x2 = x1 / scale, x2 / scale
                        y1, y2 = y1 / scale, y2 / scale
                        # Clip to image boundaries
                        x1, x2 = max(0, min(x1, orig_w)), max(0, min(x2, orig_w))
                        y1, y2 = max(0, min(y1, orig_h)), max(0, min(y2, orig_h))
                        # Create face info dictionary
                        faces.append({
                            'box': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'confidence': float(conf),
                            'keypoints': {}  # YOLOv5 doesn't provide keypoints by default
                        })
            else:
                # Process standard tensor output
                for det in detections:
                    if len(det) >= 5:
                        x1, y1, x2, y2, conf = det[:5]
                        if float(conf) >= self.confidence_threshold:
                            # Convert to original image coordinates
                            x1, x2 = x1 / scale, x2 / scale
                            y1, y2 = y1 / scale, y2 / scale
                            # Clip to image boundaries
                            x1, x2 = max(0, min(x1, orig_w)), max(0, min(x2, orig_w))
                            y1, y2 = max(0, min(y1, orig_h)), max(0, min(y2, orig_h))
                            # Create face info dictionary
                            faces.append({
                                'box': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'confidence': float(conf),
                                'keypoints': {}
                            })
        
        return faces
    
    def detect_faces(self, image):
        start_time = time.time() if self.enable_timing else None
        
        try:
            # Preprocess image
            img_tensor, image_info = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                # Use direct PyTorch model
                detections = self.model(img_tensor)
            
            # Process detections
            faces = self.postprocess_detections(detections, image_info)
            
            if self.enable_timing:
                inference_time = time.time() - start_time
                print(f"YOLOv5 inference time: {inference_time*1000:.1f}ms, detected {len(faces)} faces")
            
            return faces
            
        except Exception as e:
            print(f"Error during YOLOv5 face detection: {e}")
            return []
    
    def __call__(self, image):
        return self.detect_faces(image)


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLOv5FaceDetector(confidence_threshold=0.5, enable_timing=True)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw faces
        for face in faces:
            x, y, w, h = face['box']
            conf = face['confidence']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display
        cv2.imshow(' YOLOv5 Face Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
