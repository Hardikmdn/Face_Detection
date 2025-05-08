#!/usr/bin/env python3
import cv2
import argparse
import time
import numpy as np
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
import os

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time face recognition')
    
    parser.add_argument('--detector', type=str, default='mtcnn', choices=['mtcnn', 'yolov5'],
                        help='Face detector to use')
    parser.add_argument('--confidence', type=float, default=0.9,
                        help='Confidence threshold for face detection')
    parser.add_argument('--recognition-threshold', type=float, default=0.7,
                        help='Primary confidence threshold for face recognition')
    parser.add_argument('--unknown-threshold', type=float, default=0.5,
                        help='Minimum threshold to consider any recognition')
    parser.add_argument('--min-confidence-margin', type=float, default=0.2,
                        help='Minimum margin between top prediction and second best')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--display-fps', action='store_true',
                        help='Display FPS on the video feed')
    parser.add_argument('--display-confidence', action='store_true',
                        help='Display confidence scores')
    parser.add_argument('--display-margin', action='store_true',
                        help='Display confidence margin between top predictions')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory containing the trained models')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize face detector
    print(f"Initializing {args.detector} face detector...")
    face_detector = FaceDetector(
        detector_type=args.detector,
        confidence_threshold=args.confidence
    )
    
    # Initialize face recognizer
    print("Initializing face recognizer...")
    facenet_model_path = os.path.join(args.model_dir, 'facenet_keras.h5')
    classifier_path = os.path.join(args.model_dir, 'face_classifier_model.h5')
    
    face_recognizer = FaceRecognizer(
        model_path=facenet_model_path,
        classifier_path=classifier_path
    )
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variables for FPS calculation
    frame_count = 0
    fps = 0
    start_time = time.time()
    
    print("Starting real-time recognition. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Detect faces
        faces = face_detector.detect_faces(frame)
        
        # Process each detected face
        for face_info in faces:
            # Extract face coordinates
            x, y, width, height = face_info['box']
            
            # Extract face
            face_img = face_detector.extract_face(frame, face_info)
            
            # Recognize face with  unknown handling
            result = face_recognizer.recognize(
                face_img, 
                threshold=args.recognition_threshold,
                unknown_threshold=args.unknown_threshold,
                min_confidence_margin=args.min_confidence_margin
            )
            
            # Draw bounding box
            color = (0, 255, 0) if result['recognized'] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
            
            # Display name and confidence
            name = result['name']
            confidence = result['confidence']
            margin = result['confidence_margin']
            
            text = name
            if args.display_confidence and args.display_margin:
                text = f"{name} ({confidence:.2f}, margin: {margin:.2f})"
            elif args.display_confidence:
                text = f"{name} ({confidence:.2f})"
            elif args.display_margin:
                text = f"{name} (margin: {margin:.2f})"
            
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display FPS
        if args.display_fps:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Real-time Face Recognition', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped")

if __name__ == '__main__':
    main()
