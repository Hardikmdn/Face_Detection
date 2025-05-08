#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Add, LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize face recognition neural network architecture')
    
    parser.add_argument('--output-dir', type=str, default='metrics',
                        help='Directory to save the visualization plots')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes for the output layer')
    parser.add_argument('--show-shapes', action='store_true',
                        help='Show shapes in the model visualization')
    parser.add_argument('--show-layer-names', action='store_true',
                        help='Show layer names in the model visualization')
    
    return parser.parse_args()

def residual_block(x, units, dropout_rate=0.3, l2_reg=0.0005):
    """Create a residual block."""
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

def build_classifier(num_classes, embedding_size=2048):
    """Build the classifier model."""
    # Create a functional model for more flexibility
    inputs = Input(shape=(embedding_size,))
    
    # Initial dense layer to process the embedding
    x = Dense(1024, kernel_regularizer=l2(0.0005))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    
    # First residual block (512 units)
    x = residual_block(x, 512, dropout_rate=0.3)
    
    # Second residual block (256 units)
    x = residual_block(x, 256, dropout_rate=0.2)
    
    # Final feature layer
    x = Dense(128, kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Output layer with softmax activation
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0005))(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualize_model_architecture(model, output_dir, show_shapes=True, show_layer_names=True):
    """Visualize the model architecture."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the model visualization
    plot_model(
        model,
        to_file=os.path.join(output_dir, 'model_architecture.png'),
        show_shapes=show_shapes,
        show_layer_names=show_layer_names,
        dpi=150
    )
    
    # Print model summary
    model.summary()
    
    # Save model summary to file
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        # Redirect summary to file
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Model visualization saved to {os.path.join(output_dir, 'model_architecture.png')}")
    print(f"Model summary saved to {os.path.join(output_dir, 'model_summary.txt')}")

def create_text_visualization(model, output_dir):
    """Create a text-based visualization of the model architecture."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model layers
    layers = model.layers
    
    # Create ASCII art representation
    with open(os.path.join(output_dir, 'model_ascii.txt'), 'w') as f:
        f.write("Face Recognition Neural Network Architecture\n")
        f.write("=" * 50 + "\n\n")
        
        # Input layer
        f.write("Input Layer (2048) → Face Embedding Vector\n")
        f.write("    ↓\n")
        
        # Track the layer index
        layer_idx = 1
        
        # Process each layer
        while layer_idx < len(layers):
            layer = layers[layer_idx]
            
            if isinstance(layer, Dense) and layer_idx + 2 < len(layers) and isinstance(layers[layer_idx + 2], Dropout):
                # This is the first dense layer
                f.write(f"Dense Layer (1024) + BatchNorm + LeakyReLU + Dropout(0.4)\n")
                f.write("    ↓\n")
                layer_idx += 3  # Skip BatchNorm, LeakyReLU, and Dropout
            
            elif isinstance(layer, Dense) and layer_idx + 6 < len(layers) and isinstance(layers[layer_idx + 6], Add):
                # This is the start of a residual block
                units = layer.get_config()['units']
                dropout_rate = layers[layer_idx + 3].rate if isinstance(layers[layer_idx + 3], Dropout) else "?"
                
                f.write(f"Residual Block ({units} units):\n")
                f.write(f"    ├── Dense({units}) + BatchNorm + LeakyReLU + Dropout({dropout_rate})\n")
                f.write(f"    ├── Dense({units}) + BatchNorm\n")
                
                # Check if there's a projection layer
                if layer_idx + 8 < len(layers) and isinstance(layers[layer_idx + 6], Add):
                    prev_units = layers[layer_idx - 1].get_config()['units'] if hasattr(layers[layer_idx - 1], 'get_config') else None
                    current_units = units
                    if prev_units != current_units:
                        f.write(f"    ├── Projection: Dense({units}) + BatchNorm\n")
                
                f.write(f"    └── Add + LeakyReLU\n")
                f.write("    ↓\n")
                
                # Skip to after the residual block
                # Find the LeakyReLU after the Add layer
                for i in range(layer_idx, len(layers)):
                    if isinstance(layers[i], Add):
                        layer_idx = i + 2  # Skip Add and LeakyReLU
                        break
            
            elif isinstance(layer, Dense) and layer_idx + 2 < len(layers) and not isinstance(layers[layer_idx + 3], Dropout) and layer != layers[-1]:
                # This is the final feature layer
                f.write(f"Dense Layer (128) + BatchNorm + LeakyReLU\n")
                f.write("    ↓\n")
                layer_idx += 3  # Skip BatchNorm and LeakyReLU
            
            elif layer == layers[-1]:
                # Output layer
                output_units = layer.get_config()['units']
                f.write(f"Output Layer ({output_units}) with Softmax Activation\n")
                f.write("    ↓\n")
                f.write("Classification Result\n")
                layer_idx += 1
            
            else:
                # Skip other layers or increment if not handled
                layer_idx += 1
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Optimizer: Adam (lr=0.001)\n")
        f.write("Loss: Categorical Cross-Entropy\n")
        f.write("Metrics: Accuracy\n")
    
    print(f"ASCII model visualization saved to {os.path.join(output_dir, 'model_ascii.txt')}")

def main():
    """Main function."""
    args = parse_args()
    
    # Build the classifier model
    model = build_classifier(args.num_classes)
    
    # Visualize the model architecture
    visualize_model_architecture(
        model, 
        args.output_dir,
        show_shapes=args.show_shapes,
        show_layer_names=args.show_layer_names
    )
    
    # Create text-based visualization
    create_text_visualization(model, args.output_dir)

if __name__ == '__main__':
    main()
