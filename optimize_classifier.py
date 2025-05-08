#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import time

# Import our existing face recognition components
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from utils import load_dataset, create_model_directories, plot_training_history

class ClassifierHyperModel:
    """
    Hypermodel for tuning the face classifier neural network based on our existing architecture.
    This class adapts our current classifier architecture to support hyperparameter tuning.
    """
    
    def __init__(self, embedding_size, num_classes):
        self.embedding_size = embedding_size
        self.num_classes = num_classes
    
    def residual_block(self, x, units, dropout_rate, l2_reg):
        """
        Create a residual block with batch normalization and LeakyReLU activation.
        
        Args:
            x: Input tensor
            units: Number of units in the dense layers
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
            
        Returns:
            Output tensor
        """
        # Store the input for the residual connection
        residual = x
        
        # First dense layer
        x = layers.Dense(units, kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Second dense layer
        x = layers.Dense(units, kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        
        # If input and output dimensions don't match, use a projection shortcut
        if residual.shape[-1] != units:
            residual = layers.Dense(units, kernel_regularizer=l2(l2_reg))(residual)
            residual = layers.BatchNormalization()(residual)
        
        # Add the residual connection
        x = layers.Add()([x, residual])
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        return x
    
    def build_model(self, hp):
        """
        Build the model with hyperparameters to tune.
        
        Args:
            hp: Hyperparameters object from Keras Tuner
            
        Returns:
            Compiled Keras model
        """
        # Create a functional model for more flexibility
        inputs = layers.Input(shape=(self.embedding_size,))
        
        # Initial dense layer to process the embedding
        initial_units = hp.Int('initial_units', min_value=512, max_value=2048, step=256)
        initial_dropout = hp.Float('initial_dropout', min_value=0.2, max_value=0.5, step=0.1)
        initial_l2 = hp.Float('initial_l2', min_value=0.0001, max_value=0.001, sampling='log')
        
        x = layers.Dense(initial_units, kernel_regularizer=l2(initial_l2))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=hp.Float('leaky_alpha', min_value=0.01, max_value=0.2, step=0.01))(x)
        x = layers.Dropout(initial_dropout)(x)
        
        # First residual block
        res1_units = hp.Int('res1_units', min_value=256, max_value=768, step=128)
        res1_dropout = hp.Float('res1_dropout', min_value=0.2, max_value=0.4, step=0.05)
        res1_l2 = hp.Float('res1_l2', min_value=0.0001, max_value=0.001, sampling='log')
        
        x = self.residual_block(x, res1_units, res1_dropout, res1_l2)
        
        # Second residual block (optional)
        if hp.Boolean('use_second_residual'):
            res2_units = hp.Int('res2_units', min_value=128, max_value=512, step=128)
            res2_dropout = hp.Float('res2_dropout', min_value=0.1, max_value=0.3, step=0.05)
            res2_l2 = hp.Float('res2_l2', min_value=0.0001, max_value=0.001, sampling='log')
            
            x = self.residual_block(x, res2_units, res2_dropout, res2_l2)
        
        # Final feature layer
        final_units = hp.Int('final_units', min_value=64, max_value=256, step=64)
        final_l2 = hp.Float('final_l2', min_value=0.0001, max_value=0.001, sampling='log')
        
        x = layers.Dense(final_units, kernel_regularizer=l2(final_l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        # Output layer with softmax activation
        outputs = layers.Dense(self.num_classes, activation='softmax', 
                              kernel_regularizer=l2(0.0005))(x)
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Optimizer selection
        optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        
        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize face recognition classifier')
    
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--detector', type=str, default='mtcnn', choices=['mtcnn', 'yolov5'],
                        help='Face detector to use')
    parser.add_argument('--confidence', type=float, default=0.9,
                        help='Confidence threshold for face detection')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--max-trials', type=int, default=20,
                        help='Maximum number of hyperparameter trials')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs per trial')
    parser.add_argument('--tuner', type=str, default='hyperband',
                        choices=['random', 'hyperband', 'bayesian'],
                        help='Tuner algorithm to use')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save the optimized model')
    
    return parser.parse_args()

def main():
    """Main function for optimizing the face recognition classifier."""
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
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    face_images, labels = load_dataset(
        dataset_path=args.dataset,
        face_detector=face_detector
    )
    
    print(f"Loaded {len(face_images)} images with {len(np.unique(labels))} unique labels")
    
    # Initialize face recognizer for embedding generation
    print("Initializing face recognizer for embedding generation...")
    face_recognizer = FaceRecognizer()
    
    # Generate embeddings for all faces
    print("Generating embeddings for training faces...")
    embeddings = []
    for face in face_images:
        embedding = face_recognizer.get_embedding(face)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Convert to categorical
    categorical_labels = to_categorical(encoded_labels, num_classes=num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, categorical_labels, test_size=args.test_size, 
        random_state=42, stratify=categorical_labels
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Import Keras Tuner (import here to avoid errors if not installed)
    try:
        import keras_tuner as kt
    except ImportError:
        print("Keras Tuner not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "keras-tuner"])
        import keras_tuner as kt
    
    # Create hypermodel
    hypermodel = ClassifierHyperModel(
        embedding_size=embeddings.shape[1],
        num_classes=num_classes
    )
    
    # Create tuner directory
    tuner_dir = os.path.join(args.output_dir, 'tuner_results')
    os.makedirs(tuner_dir, exist_ok=True)
    
    # Create tuner based on selected algorithm
    print(f"Creating {args.tuner} tuner...")
    if args.tuner == 'random':
        tuner = kt.RandomSearch(
            hypermodel.build_model,
            objective='val_accuracy',
            max_trials=args.max_trials,
            directory=tuner_dir,
            project_name='face_classifier_optimization'
        )
    elif args.tuner == 'hyperband':
        tuner = kt.Hyperband(
            hypermodel.build_model,
            objective='val_accuracy',
            max_epochs=args.epochs,
            factor=3,
            directory=tuner_dir,
            project_name='face_classifier_optimization'
        )
    elif args.tuner == 'bayesian':
        tuner = kt.BayesianOptimization(
            hypermodel.build_model,
            objective='val_accuracy',
            max_trials=args.max_trials,
            directory=tuner_dir,
            project_name='face_classifier_optimization'
        )
    
    # Print search space summary
    print("\nSearch space summary:")
    tuner.search_space_summary()
    
    # Create callbacks for training
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Start hyperparameter search
    print(f"\nStarting hyperparameter search with {args.max_trials} trials...")
    start_time = time.time()
    
    tuner.search(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )
    
    search_time = time.time() - start_time
    print(f"Hyperparameter search completed in {search_time:.2f} seconds")
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nBest hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    
    # Build model with best hyperparameters
    print("\nBuilding model with best hyperparameters...")
    best_model = tuner.hypermodel.build(best_hps)
    
    # Train the best model
    print("\nTraining the model with best hyperparameters...")
    history = best_model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )
    
    # Evaluate the best model
    print("\nEvaluating the best model...")
    y_pred_prob = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    # Save the best model
    model_path = os.path.join(args.output_dir, 'optimized_face_classifier_model.h5')
    best_model.save(model_path)
    print(f"Saved optimized model to {model_path}")
    
    # Save the label encoder
    label_encoder_path = os.path.join(args.output_dir, 'optimized_label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder to {label_encoder_path}")
    
    # Save the best hyperparameters
    hyperparams_path = os.path.join(args.output_dir, 'best_hyperparameters.pkl')
    with open(hyperparams_path, 'wb') as f:
        pickle.dump(best_hps.values, f)
    print(f"Saved best hyperparameters to {hyperparams_path}")
    
    # Plot training history
    print("Plotting training history...")
    fig = plot_training_history(history.history)
    
    # Save the plot
    plot_path = os.path.join(args.output_dir, 'optimized_training_history.png')
    fig.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    
    print("\nOptimization completed successfully!")
    print(f"Optimized model saved to {model_path}")
    print(f"To use this model, set the classifier_path parameter when initializing FaceRecognizer")

if __name__ == '__main__':
    main()
