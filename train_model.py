#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from utils import load_dataset, split_dataset, plot_training_history, create_model_directories, augment_dataset
import pickle
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train face recognition model')
    
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--augmented-dataset', type=str, default='augmented_dataset',
                        help='Path to save the augmented dataset')
    parser.add_argument('--detector', type=str, default='mtcnn', choices=['mtcnn', 'yolov5'],
                        help='Face detector to use')
    parser.add_argument('--confidence', type=float, default=0.9,
                        help='Confidence threshold for face detection')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--augment', action='store_true',
                        help='Augment the dataset')
    parser.add_argument('--samples-per-class', type=int, default=100,
                        help='Number of samples to generate per class when augmenting')
    parser.add_argument('--use-augmented', action='store_true',
                        help='Use the augmented dataset for training')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save the trained model')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create model directories
    model_dirs = create_model_directories()
    
    # Initialize face detector
    print(f"Initializing {args.detector} face detector...")
    face_detector = FaceDetector(
        detector_type=args.detector,
        confidence_threshold=args.confidence
    )
    
    # Determine dataset path
    dataset_path = args.dataset
    if args.use_augmented:
        dataset_path = args.augmented_dataset
        print(f"Using augmented dataset from {dataset_path}")
    
    # Augment dataset if requested
    if args.augment:
        print(f"Augmenting dataset from {args.dataset} to {args.augmented_dataset}...")
        augment_dataset(
            dataset_path=args.dataset,
            output_path=args.augmented_dataset,
            samples_per_class=args.samples_per_class,
            face_detector=face_detector
        )
        
        if args.use_augmented:
            dataset_path = args.augmented_dataset
            print(f"Using augmented dataset for training")
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    face_images, labels = load_dataset(
        dataset_path=dataset_path,
        face_detector=face_detector
    )
    
    print(f"Loaded {len(face_images)} images with {len(np.unique(labels))} unique labels")
    
    # Initialize face recognizer
    print("Initializing face recognizer...")
    face_recognizer = FaceRecognizer(
        classifier_path=os.path.join(args.output_dir, 'face_classifier_model.h5')
    )
    
    # Train the model
    print("Training face recognition model...")
    start_time = time.time()
    
    results = face_recognizer.train(
        face_images=face_images,
        labels=labels,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Display results
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    for class_name, metrics in results['report'].items():
        if isinstance(metrics, dict):
            print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1-score']:.4f}")
    
    # Plot training history
    print("Plotting training history...")
    fig = plot_training_history(results['history'])
    
    # Save the plot
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    
    # Save training results
    results_path = os.path.join(args.output_dir, 'training_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Training results saved to {results_path}")
    
    print("\nTraining completed successfully!")
    print(f"Model saved to {face_recognizer.classifier_path}")
    print(f"Label encoder saved to {face_recognizer.label_encoder_path}")

if __name__ == '__main__':
    main()
