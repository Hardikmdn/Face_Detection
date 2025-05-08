#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import pandas as pd
from matplotlib.gridspec import GridSpec

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize face recognition metrics')
    
    parser.add_argument('--results-file', type=str, default='models/training_results.pkl',
                        help='Path to the training results pickle file')
    parser.add_argument('--output-dir', type=str, default='metrics',
                        help='Directory to save the visualization plots')
    parser.add_argument('--unknown-class', type=str, default='Unknown',
                        help='Label for the unknown class (if available)')
    
    return parser.parse_args()

def load_results(results_file):
    """Load training results from pickle file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return results

def filter_class_from_results(results, class_to_filter):
    """Filter out a specific class from the results."""
    # If no results or no report, return as is
    if not results or 'report' not in results:
        return results
    
    # Make a deep copy to avoid modifying the original
    filtered_results = dict(results)
    
    # Filter out the class from the report if it exists
    if class_to_filter in filtered_results['report']:
        filtered_report = dict(filtered_results['report'])
        filtered_report.pop(class_to_filter, None)
        filtered_results['report'] = filtered_report
    
    # Filter out the class from classes list if it exists
    if 'classes' in filtered_results and class_to_filter in filtered_results['classes']:
        filtered_classes = list(filtered_results['classes'])
        class_index = filtered_classes.index(class_to_filter)
        filtered_classes.remove(class_to_filter)
        filtered_results['classes'] = filtered_classes
        
        # If y_true and y_pred exist, filter out the class from them too
        if 'y_true' in filtered_results and 'y_pred' in filtered_results:
            y_true = np.array(filtered_results['y_true'])
            y_pred = np.array(filtered_results['y_pred'])
            
            # Create mask for entries that are not the class to filter
            mask = y_true != class_index
            
            # Apply mask to filter out the class
            filtered_results['y_true'] = y_true[mask]
            filtered_results['y_pred'] = y_pred[mask]
    
    return filtered_results

def plot_training_history(history, output_dir):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_classification_metrics(report, output_dir):
    """Plot precision, recall, and F1-score for each class."""
    # Convert classification report to DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Filter out accuracy, macro avg, and weighted avg
    df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot metrics
    df.plot(y=['precision', 'recall', 'f1-score'], kind='bar', ax=ax)
    ax.set_title('Classification Metrics by Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'classification_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def calculate_false_rates(y_true, y_pred, unknown_class=None):
    """
    Calculate false positive and false negative rates.
    
    If unknown_class is provided, calculates:
    - False Positive Rate: Rate of unknown faces incorrectly identified as known
    - False Negative Rate: Rate of known faces incorrectly identified as unknown
    
    Otherwise, calculates general false positive and negative rates.
    """
    if unknown_class and unknown_class in np.unique(y_true):
        # Create binary classification: known vs unknown
        y_true_binary = np.array([1 if label != unknown_class else 0 for label in y_true])
        y_pred_binary = np.array([1 if label != unknown_class else 0 for label in y_pred])
        
        # Calculate rates
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Total unknown samples
        unknown_total = np.sum(y_true_binary == 0)
        # Total known samples
        known_total = np.sum(y_true_binary == 1)
        
        # Calculate rates
        fpr = fp / unknown_total if unknown_total > 0 else 0
        fnr = fn / known_total if known_total > 0 else 0
        
        return {
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'false_positives': fp,
            'false_negatives': fn,
            'unknown_samples': unknown_total,
            'known_samples': known_total
        }
    else:
        # For general multi-class case
        n_classes = len(np.unique(y_true))
        
        # Initialize arrays to store FPR and FNR for each class
        fpr_per_class = []
        fnr_per_class = []
        class_names = []
        
        # Calculate FPR and FNR for each class
        for cls in np.unique(y_true):
            # Binary classification: this class vs. rest
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            # Calculate rates
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            
            # Calculate rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            fpr_per_class.append(fpr)
            fnr_per_class.append(fnr)
            class_names.append(cls)
        
        # Calculate average rates
        avg_fpr = np.mean(fpr_per_class)
        avg_fnr = np.mean(fnr_per_class)
        
        return {
            'false_positive_rate': avg_fpr,
            'false_negative_rate': avg_fnr,
            'false_positive_rate_per_class': dict(zip(class_names, fpr_per_class)),
            'false_negative_rate_per_class': dict(zip(class_names, fnr_per_class))
        }

def plot_false_rates(false_rates, output_dir, unknown_class=None):
    """Plot false positive and false negative rates."""
    if unknown_class and 'unknown_samples' in false_rates:
        # Create figure for unknown vs known case
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        metrics = ['False Positive Rate', 'False Negative Rate']
        values = [false_rates['false_positive_rate'], false_rates['false_negative_rate']]
        
        bars = ax.bar(metrics, values, color=['#ff9999', '#66b3ff'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom')
        
        ax.set_ylim(0, max(values) * 1.2 or 0.1)  # Set y-limit with some padding
        ax.set_title(f'False Rates for Unknown vs Known Classification')
        ax.set_ylabel('Rate')
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add explanation text
        explanation = (
            f"False Positive Rate: {false_rates['false_positives']} out of {false_rates['unknown_samples']} "
            f"unknown faces incorrectly identified as known\n"
            f"False Negative Rate: {false_rates['false_negatives']} out of {false_rates['known_samples']} "
            f"known faces incorrectly identified as unknown"
        )
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=9, 
                   bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    else:
        # Create figure for per-class case
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Get class names and rates
        classes = list(false_rates['false_positive_rate_per_class'].keys())
        fpr_values = list(false_rates['false_positive_rate_per_class'].values())
        fnr_values = list(false_rates['false_negative_rate_per_class'].values())
        
        # Sort by false positive rate
        sorted_indices = np.argsort(fpr_values)[::-1]
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_fpr = [fpr_values[i] for i in sorted_indices]
        sorted_fnr = [fnr_values[i] for i in sorted_indices]
        
        # Plot FPR
        bars1 = ax1.bar(sorted_classes, sorted_fpr, color='#ff9999')
        ax1.set_title('False Positive Rate by Class')
        ax1.set_ylabel('False Positive Rate')
        ax1.set_ylim(0, max(sorted_fpr) * 1.2 or 0.1)
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax1.axhline(y=false_rates['false_positive_rate'], color='r', linestyle='--', 
                   label=f'Average: {false_rates["false_positive_rate"]:.2%}')
        ax1.legend()
        
        # Plot FNR
        bars2 = ax2.bar(sorted_classes, sorted_fnr, color='#66b3ff')
        ax2.set_title('False Negative Rate by Class')
        ax2.set_ylabel('False Negative Rate')
        ax2.set_ylim(0, max(sorted_fnr) * 1.2 or 0.1)
        ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax2.axhline(y=false_rates['false_negative_rate'], color='b', linestyle='--',
                   label=f'Average: {false_rates["false_negative_rate"]:.2%}')
        ax2.legend()
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'false_rates.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_summary_dashboard(results, false_rates, output_dir):
    """Create a summary dashboard with key metrics."""
    # Create figure with grid layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Overall accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('equal')
    accuracy = results['accuracy']
    ax1.pie([accuracy, 1-accuracy], labels=['Correct', 'Incorrect'], 
           autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'],
           wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    ax1.set_title('Overall Accuracy')
    
    # False rates
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['False Positive Rate', 'False Negative Rate']
    values = [false_rates['false_positive_rate'], false_rates['false_negative_rate']]
    bars = ax2.bar(metrics, values, color=['#ff9999', '#66b3ff'])
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2%}', ha='center', va='bottom')
    ax2.set_ylim(0, max(values) * 1.2 or 0.1)
    ax2.set_title('Error Rates')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Class metrics
    ax3 = fig.add_subplot(gs[1, :])
    
    # Get class metrics
    df = pd.DataFrame(results['report']).transpose()
    df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    
    # Sort by F1-score
    df = df.sort_values('f1-score', ascending=False)
    
    # Plot metrics
    df.plot(y=['precision', 'recall', 'f1-score'], kind='bar', ax=ax3)
    ax3.set_title('Performance Metrics by Class')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Score')
    ax3.set_ylim([0, 1.05])
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Add title
    fig.suptitle('Face Recognition System Performance Dashboard', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function for visualizing metrics."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    try:
        results = load_results(args.results_file)
        
        # Filter out Yayun from the results
        print("Filtering out 'yayun' from the results...")
        results = filter_class_from_results(results, 'yayun')
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results_file}")
        print("Creating sample data for demonstration purposes...")
        # Create sample data for demonstration
        results = create_sample_data()
    
    # Extract data
    history = results.get('history', None)
    classes = results.get('classes', [])
    report = results.get('report', {})
    
    # Check if we have history data
    has_history = history is not None and 'accuracy' in history and 'val_accuracy' in history
    
    # Convert class indices to class names for confusion matrix
    y_true = np.argmax(results.get('y_test', []), axis=1) if 'y_test' in results else None
    y_pred = np.argmax(results.get('y_pred', []), axis=1) if 'y_pred' in results else None
    
    # If y_true and y_pred are not in results, try to reconstruct from report
    has_prediction_data = False
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: Test data not found in results. Reconstructing from report...")
        # This is a simplified reconstruction and may not be accurate
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            y_true = []
            y_pred = []
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    y_true.extend([i] * cm[i][j])
                    y_pred.extend([j] * cm[i][j])
            has_prediction_data = len(y_true) > 0
        else:
            # Create synthetic data from the classification report
            y_true, y_pred, classes = create_synthetic_data_from_report(report)
            has_prediction_data = len(y_true) > 0
    
    # Plot training history if available
    if has_history:
        print("Plotting training history...")
        plot_training_history(history, args.output_dir)
    else:
        print("Warning: Training history not found in results. Skipping history plot.")
    
    # Plot confusion matrix if we have the data
    if has_prediction_data and len(classes) > 0:
        print("Plotting confusion matrix...")
        try:
            plot_confusion_matrix(y_true, y_pred, classes, args.output_dir)
        except Exception as e:
            print(f"Error plotting confusion matrix: {str(e)}")
    else:
        print("Warning: Insufficient data for confusion matrix. Skipping.")
    
    # Plot classification metrics if report exists
    if report and len(report) > 0:
        print("Plotting classification metrics...")
        try:
            plot_classification_metrics(report, args.output_dir)
        except Exception as e:
            print(f"Error plotting classification metrics: {str(e)}")
    else:
        print("Warning: Classification report not found. Skipping metrics plot.")
    
    # Calculate and plot false rates
    if has_prediction_data and len(classes) > 0:
        print("Calculating false rates...")
        try:
            # Convert numeric labels to class names
            y_true_names = [classes[i] for i in y_true]
            y_pred_names = [classes[i] for i in y_pred]
            
            # Calculate false rates
            false_rates = calculate_false_rates(y_true_names, y_pred_names, args.unknown_class)
            
            # Plot false rates
            print("Plotting false rates...")
            plot_false_rates(false_rates, args.output_dir, args.unknown_class)
            
            # Create summary dashboard
            print("Creating summary dashboard...")
            create_summary_dashboard(results, false_rates, args.output_dir)
        except Exception as e:
            print(f"Error calculating/plotting false rates: {str(e)}")
    else:
        print("Warning: Insufficient data for false rate analysis. Skipping.")
    
    print(f"All visualizations saved to {args.output_dir}")
    print("\nKey metrics summary:")
    if 'accuracy' in results:
        print(f"Overall accuracy: {results['accuracy']:.2%}")
    else:
        print("Overall accuracy: Not available")
    
    if 'macro avg' in report:
        print(f"Macro-average precision: {report['macro avg']['precision']:.2%}")
        print(f"Macro-average recall: {report['macro avg']['recall']:.2%}")
        print(f"Macro-average F1-score: {report['macro avg']['f1-score']:.2%}")
    else:
        print("Macro-average metrics: Not available")
    
    if has_prediction_data and args.unknown_class in classes:
        try:
            false_rates = calculate_false_rates(y_true_names, y_pred_names, args.unknown_class)
            print(f"False positive rate (unknown as known): {false_rates['false_positive_rate']:.2%}")
            print(f"False negative rate (known as unknown): {false_rates['false_negative_rate']:.2%}")
        except Exception as e:
            print(f"Error calculating false rates: {str(e)}")

def create_sample_data():
    """Create sample data for demonstration when no results file is available."""
    # Create sample history
    history = {
        'accuracy': [0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracy': [0.45, 0.55, 0.65, 0.75, 0.78, 0.82],
        'loss': [0.8, 0.6, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.9, 0.7, 0.5, 0.35, 0.3, 0.25]
    }
    
    # Create sample classes (without Yayun)
    classes = ['Person1', 'Person2', 'Person3', 'Person4', 'Unknown']
    
    # Create sample report (without Yayun)
    report = {
        'Person1': {'precision': 0.92, 'recall': 0.88, 'f1-score': 0.90, 'support': 50},
        'Person2': {'precision': 0.85, 'recall': 0.90, 'f1-score': 0.87, 'support': 50},
        'Person3': {'precision': 0.95, 'recall': 0.92, 'f1-score': 0.93, 'support': 50},
        'Person4': {'precision': 0.88, 'recall': 0.86, 'f1-score': 0.87, 'support': 50},
        'Unknown': {'precision': 0.75, 'recall': 0.80, 'f1-score': 0.77, 'support': 50},
        'accuracy': 0.87,
        'macro avg': {'precision': 0.87, 'recall': 0.87, 'f1-score': 0.87, 'support': 250},
        'weighted avg': {'precision': 0.87, 'recall': 0.87, 'f1-score': 0.87, 'support': 250}
    }
    
    # Create sample confusion matrix
    cm = np.array([
        [44, 3, 1, 1, 1],
        [2, 45, 1, 1, 1],
        [1, 1, 46, 1, 1],
        [1, 1, 1, 43, 4],
        [2, 3, 1, 4, 40]
    ])
    
    # Create sample y_true and y_pred
    y_true = []
    y_pred = []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            y_true.extend([i] * cm[i][j])
            y_pred.extend([j] * cm[i][j])
    
    return {
        'history': history,
        'classes': classes,
        'report': report,
        'confusion_matrix': cm,
        'accuracy': 0.87,
        'y_true': y_true,
        'y_pred': y_pred
    }

def create_synthetic_data_from_report(report):
    """Create synthetic test data from classification report."""
    if not report or 'macro avg' not in report:
        # Return empty data if report is invalid
        return [], [], []
    
    # Extract class names (excluding metrics)
    class_names = [k for k in report.keys() 
                  if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    if not class_names:
        return [], [], []
    
    # Create synthetic data based on support and metrics
    y_true = []
    y_pred = []
    
    for i, class_name in enumerate(class_names):
        if 'support' not in report[class_name]:
            continue
            
        support = int(report[class_name]['support'])
        recall = float(report[class_name]['recall'])
        
        # True positives for this class
        true_positives = int(support * recall)
        false_negatives = support - true_positives
        
        # Add true positives (correctly predicted)
        y_true.extend([i] * true_positives)
        y_pred.extend([i] * true_positives)
        
        # Add false negatives (incorrectly predicted as other classes)
        for j, other_class in enumerate(class_names):
            if i != j:
                # Distribute false negatives among other classes
                false_per_class = max(1, false_negatives // (len(class_names) - 1))
                if false_negatives > 0:
                    y_true.extend([i] * false_per_class)
                    y_pred.extend([j] * false_per_class)
                    false_negatives -= false_per_class
    
    return np.array(y_true), np.array(y_pred), class_names

if __name__ == '__main__':
    main()
