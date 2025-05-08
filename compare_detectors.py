#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from face_detection import FaceDetector
import argparse
from tqdm import tqdm
import pandas as pd
from matplotlib.gridspec import GridSpec

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare face detector performance')
    
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--output-dir', type=str, default='detector_comparison',
                        help='Directory to save the comparison results')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images to use for comparison')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for face detection')
    
    return parser.parse_args()

def load_test_images(dataset_path, max_images=100):
    """Load test images from dataset."""
    images = []
    paths = []
    
    print(f"Loading test images from {dataset_path}...")
    
    # Walk through all subdirectories
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_dir):
            continue
        
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
            
            images.append(img)
            paths.append(img_path)
            
            if len(images) >= max_images:
                break
        
        if len(images) >= max_images:
            break
    
    print(f"Loaded {len(images)} test images")
    return images, paths

def evaluate_detector(detector, images, paths):
    """Evaluate a face detector on a set of images."""
    results = []
    
    for i, (img, path) in enumerate(tqdm(zip(images, paths), total=len(images), desc=f"Evaluating {detector.detector_type}")):
        start_time = time.time()
        
        # Detect faces
        faces = detector.detect_faces(img)
        
        # Calculate metrics
        detection_time = time.time() - start_time
        num_faces = len(faces)
        
        # Store results
        results.append({
            'detector': detector.detector_type,
            'image_path': path,
            'num_faces': num_faces,
            'detection_time': detection_time,
            'faces': faces
        })
    
    return results

def plot_detection_time_comparison(mtcnn_results, yolo_results, output_dir):
    """Plot detection time comparison."""
    mtcnn_times = [r['detection_time'] for r in mtcnn_results]
    yolo_times = [r['detection_time'] for r in yolo_results]
    
    # Calculate statistics
    mtcnn_avg = np.mean(mtcnn_times)
    yolo_avg = np.mean(yolo_times)
    mtcnn_std = np.std(mtcnn_times)
    yolo_std = np.std(yolo_times)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    detectors = ['MTCNN', 'YOLOv5']
    avg_times = [mtcnn_avg, yolo_avg]
    std_times = [mtcnn_std, yolo_std]
    
    bars = ax.bar(detectors, avg_times, yerr=std_times, capsize=10, 
                 color=['#3498db', '#2ecc71'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{height:.3f}s', ha='center', va='bottom')
    
    # Add labels and title
    ax.set_ylabel('Detection Time (seconds)')
    ax.set_title('Average Face Detection Time Comparison')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add speedup text
    speedup = mtcnn_avg / yolo_avg if yolo_avg > 0 else float('inf')
    plt.figtext(0.5, 0.01, f"YOLOv5 is {speedup:.2f}x faster than MTCNN", 
               ha='center', fontsize=12, bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'detection_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_detection_count_comparison(mtcnn_results, yolo_results, output_dir):
    """Plot detection count comparison."""
    # Count images with different detection results
    total_images = len(mtcnn_results)
    same_count = sum(1 for m, y in zip(mtcnn_results, yolo_results) 
                    if m['num_faces'] == y['num_faces'])
    mtcnn_more = sum(1 for m, y in zip(mtcnn_results, yolo_results) 
                    if m['num_faces'] > y['num_faces'])
    yolo_more = sum(1 for m, y in zip(mtcnn_results, yolo_results) 
                   if m['num_faces'] < y['num_faces'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot pie chart
    labels = ['Same detections', 'MTCNN detected more', 'YOLOv5 detected more']
    sizes = [same_count, mtcnn_more, yolo_more]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    explode = (0.1, 0, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.set_title('Face Detection Count Comparison')
    
    # Plot total detections bar chart
    mtcnn_total = sum(r['num_faces'] for r in mtcnn_results)
    yolo_total = sum(r['num_faces'] for r in yolo_results)
    
    detectors = ['MTCNN', 'YOLOv5']
    total_detections = [mtcnn_total, yolo_total]
    
    bars = ax2.bar(detectors, total_detections, color=['#3498db', '#2ecc71'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    ax2.set_ylabel('Total Faces Detected')
    ax2.set_title('Total Face Detections')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'detection_count_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_detection_confidence_distribution(mtcnn_results, yolo_results, output_dir):
    """Plot detection confidence distribution."""
    # Extract confidence scores
    mtcnn_confidences = []
    yolo_confidences = []
    
    for result in mtcnn_results:
        for face in result['faces']:
            if 'confidence' in face:
                mtcnn_confidences.append(face['confidence'])
    
    for result in yolo_results:
        for face in result['faces']:
            if 'confidence' in face:
                yolo_confidences.append(face['confidence'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histograms
    bins = np.linspace(0, 1, 20)
    ax.hist(mtcnn_confidences, bins=bins, alpha=0.5, label='MTCNN', color='#3498db')
    ax.hist(yolo_confidences, bins=bins, alpha=0.5, label='YOLOv5', color='#2ecc71')
    
    # Add labels and title
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Face Detection Confidence Score Distribution')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics
    mtcnn_avg = np.mean(mtcnn_confidences) if mtcnn_confidences else 0
    yolo_avg = np.mean(yolo_confidences) if yolo_confidences else 0
    
    plt.figtext(0.5, 0.01, 
               f"Average confidence: MTCNN={mtcnn_avg:.3f}, YOLOv5={yolo_avg:.3f}", 
               ha='center', fontsize=12, bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_performance_dashboard(mtcnn_results, yolo_results, output_dir):
    """Create a comprehensive performance dashboard."""
    # Calculate metrics
    mtcnn_times = [r['detection_time'] for r in mtcnn_results]
    yolo_times = [r['detection_time'] for r in yolo_results]
    
    mtcnn_avg_time = np.mean(mtcnn_times)
    yolo_avg_time = np.mean(yolo_times)
    
    mtcnn_total_faces = sum(r['num_faces'] for r in mtcnn_results)
    yolo_total_faces = sum(r['num_faces'] for r in yolo_results)
    
    mtcnn_confidences = [face['confidence'] for r in mtcnn_results for face in r['faces'] if 'confidence' in face]
    yolo_confidences = [face['confidence'] for r in yolo_results for face in r['faces'] if 'confidence' in face]
    
    mtcnn_avg_conf = np.mean(mtcnn_confidences) if mtcnn_confidences else 0
    yolo_avg_conf = np.mean(yolo_confidences) if yolo_confidences else 0
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Detection time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    detectors = ['MTCNN', 'YOLOv5']
    avg_times = [mtcnn_avg_time, yolo_avg_time]
    
    bars1 = ax1.bar(detectors, avg_times, color=['#3498db', '#2ecc71'])
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}s', ha='center', va='bottom')
    
    ax1.set_ylabel('Detection Time (seconds)')
    ax1.set_title('Average Detection Time')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Total detections comparison
    ax2 = fig.add_subplot(gs[0, 1])
    total_detections = [mtcnn_total_faces, yolo_total_faces]
    
    bars2 = ax2.bar(detectors, total_detections, color=['#3498db', '#2ecc71'])
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    ax2.set_ylabel('Total Faces Detected')
    ax2.set_title('Total Face Detections')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Confidence distribution
    ax3 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 20)
    ax3.hist(mtcnn_confidences, bins=bins, alpha=0.5, label='MTCNN', color='#3498db')
    ax3.hist(yolo_confidences, bins=bins, alpha=0.5, label='YOLOv5', color='#2ecc71')
    
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence Score Distribution')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Performance metrics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Calculate additional metrics
    speedup = mtcnn_avg_time / yolo_avg_time if yolo_avg_time > 0 else float('inf')
    detection_diff = yolo_total_faces - mtcnn_total_faces
    detection_diff_percent = (detection_diff / mtcnn_total_faces * 100) if mtcnn_total_faces > 0 else float('inf')
    
    # Create table data
    table_data = [
        ['Metric', 'MTCNN', 'YOLOv5', 'Difference'],
        ['Avg. Detection Time (s)', f'{mtcnn_avg_time:.3f}', f'{yolo_avg_time:.3f}', f'{speedup:.2f}x faster'],
        ['Total Faces Detected', str(mtcnn_total_faces), str(yolo_total_faces), f'{detection_diff:+d} ({detection_diff_percent:+.1f}%)'],
        ['Avg. Confidence Score', f'{mtcnn_avg_conf:.3f}', f'{yolo_avg_conf:.3f}', f'{yolo_avg_conf - mtcnn_avg_conf:+.3f}']
    ]
    
    # Create table
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#f0f0f0')
                cell.set_text_props(weight='bold')
            elif j == 3:  # Difference column
                if i == 1:  # Detection time row
                    cell.set_facecolor('#d8f0d8')  # Green if YOLOv5 is faster
                elif i == 2:  # Total faces row
                    cell.set_facecolor('#d8f0d8' if detection_diff > 0 else '#f0d8d8')
                elif i == 3:  # Confidence score row
                    cell.set_facecolor('#d8f0d8' if (yolo_avg_conf - mtcnn_avg_conf) > 0 else '#f0d8d8')
    
    # Add title
    fig.suptitle('Face Detector Performance Comparison: MTCNN vs YOLOv5', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualize_sample_detections(images, mtcnn_results, yolo_results, output_dir, num_samples=5):
    """Visualize sample detections from both detectors."""
    # Select random samples
    if len(images) <= num_samples:
        sample_indices = range(len(images))
    else:
        sample_indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        img = images[idx].copy()
        mtcnn_result = mtcnn_results[idx]
        yolo_result = yolo_results[idx]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Draw MTCNN detections
        mtcnn_img = img.copy()
        for face in mtcnn_result['faces']:
            if 'box' in face:
                x, y, w, h = face['box']
                conf = face.get('confidence', 0)
                cv2.rectangle(mtcnn_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(mtcnn_img, f"{conf:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw YOLOv5 detections
        yolo_img = img.copy()
        for face in yolo_result['faces']:
            if 'box' in face:
                x, y, w, h = face['box']
                conf = face.get('confidence', 0)
                cv2.rectangle(yolo_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(yolo_img, f"{conf:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert BGR to RGB for matplotlib
        mtcnn_img = cv2.cvtColor(mtcnn_img, cv2.COLOR_BGR2RGB)
        yolo_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
        
        # Display images
        ax1.imshow(mtcnn_img)
        ax1.set_title(f"MTCNN: {len(mtcnn_result['faces'])} faces, {mtcnn_result['detection_time']:.3f}s")
        ax1.axis('off')
        
        ax2.imshow(yolo_img)
        ax2.set_title(f"YOLOv5: {len(yolo_result['faces'])} faces, {yolo_result['detection_time']:.3f}s")
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'samples', f'sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    """Main function for comparing face detectors."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test images
    images, paths = load_test_images(args.dataset, args.num_images)
    
    if not images:
        print("Error: No images found in the dataset")
        return
    
    # Initialize detectors
    mtcnn_detector = FaceDetector(
        detector_type='mtcnn',
        confidence_threshold=args.confidence,
        verbose=True
    )
    
    yolo_detector = FaceDetector(
        detector_type='yolov5',
        confidence_threshold=args.confidence,
        verbose=True
    )
    
    # Evaluate detectors
    mtcnn_results = evaluate_detector(mtcnn_detector, images, paths)
    yolo_results = evaluate_detector(yolo_detector, images, paths)
    
    # Plot comparison results
    print("Generating comparison visualizations...")
    
    plot_detection_time_comparison(mtcnn_results, yolo_results, args.output_dir)
    plot_detection_count_comparison(mtcnn_results, yolo_results, args.output_dir)
    plot_detection_confidence_distribution(mtcnn_results, yolo_results, args.output_dir)
    create_performance_dashboard(mtcnn_results, yolo_results, args.output_dir)
    visualize_sample_detections(images, mtcnn_results, yolo_results, args.output_dir)
    
    print(f"All visualizations saved to {args.output_dir}")
    
    # Print summary
    mtcnn_avg_time = np.mean([r['detection_time'] for r in mtcnn_results])
    yolo_avg_time = np.mean([r['detection_time'] for r in yolo_results])
    speedup = mtcnn_avg_time / yolo_avg_time if yolo_avg_time > 0 else float('inf')
    
    mtcnn_total_faces = sum(r['num_faces'] for r in mtcnn_results)
    yolo_total_faces = sum(r['num_faces'] for r in yolo_results)
    
    print("\nDetector Performance Summary:")
    print(f"MTCNN: {mtcnn_avg_time:.3f}s avg detection time, {mtcnn_total_faces} total faces detected")
    print(f"YOLOv5: {yolo_avg_time:.3f}s avg detection time, {yolo_total_faces} total faces detected")
    print(f"YOLOv5 is {speedup:.2f}x faster than MTCNN")

if __name__ == '__main__':
    main()
