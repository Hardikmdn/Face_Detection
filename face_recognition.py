import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization, Add, LeakyReLU, Lambda
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class FaceRecognizer:
    def __init__(self, model_path=None, classifier_path=None):
        # Set default paths for classifier
        if classifier_path is None:
            classifier_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          'models', 'face_classifier_model.h5')
        
        self.classifier_path = classifier_path
        self.label_encoder_path = os.path.join(os.path.dirname(os.path.abspath(classifier_path)), 
                                              'label_encoder.pkl')
        
        # Initialize InceptionV3 as the embedding model
        print("Loading InceptionV3 model for face embeddings...")
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
        self.embedding_model = Model(inputs=base_model.input, 
                                    outputs=GlobalAveragePooling2D()(base_model.output))
        
        # Load classifier if it exists
        self.classifier = None
        self.label_encoder = None
        self.load_classifier()
    
    def load_classifier(self):
        if os.path.exists(self.classifier_path):
            try:
                self.classifier = load_model(self.classifier_path)
                print(f"Loaded classifier model from {self.classifier_path}")
                
                # Load label encoder
                if os.path.exists(self.label_encoder_path):
                    with open(self.label_encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                    print(f"Loaded label encoder from {self.label_encoder_path}")
                else:
                    print(f"Warning: Label encoder not found at {self.label_encoder_path}")
            except Exception as e:
                print(f"Error loading classifier: {str(e)}")
        else:
            print(f"Classifier model not found at {self.classifier_path}")
    
    def preprocess_face(self, face_img):
        # Convert BGR to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize to expected size
        face_img = cv2.resize(face_img, (160, 160))
        
        # Preprocess for InceptionV3
        face_img = preprocess_input(face_img)
        
        # Expand dimensions for model input
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    
    def get_embedding(self, face_img):
        # Preprocess face
        processed_face = self.preprocess_face(face_img)
        
        # Generate embedding
        embedding = self.embedding_model.predict(processed_face)[0]
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
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
    
    def build_classifier(self, num_classes, embedding_size=2048):
        # Create a functional model for more flexibility
        inputs = Input(shape=(embedding_size,))
        
        # Initial dense layer to process the embedding
        x = Dense(1024, kernel_regularizer=l2(0.0005))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.4)(x)
        
        # First residual block (512 units)
        x = self.residual_block(x, 512, dropout_rate=0.3)
        
        # Second residual block (256 units)
        x = self.residual_block(x, 256, dropout_rate=0.2)
        
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
    
    def train(self, face_images, labels, test_size=0.2, epochs=100, batch_size=32):
        # Generate embeddings for all faces
        print("Generating embeddings for training faces...")
        embeddings = []
        for face in face_images:
            embedding = self.get_embedding(face)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        # Convert to categorical
        categorical_labels = to_categorical(encoded_labels, num_classes=num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, categorical_labels, test_size=test_size, random_state=42, stratify=categorical_labels
        )
        
        # Build  classifier
        print(f"Building  classifier for {num_classes} classes...")
        self.classifier = self.build_classifier(num_classes)
        
        # Create callbacks for better training
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model with callbacks
        print("Training  classifier...")
        history = self.classifier.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )
        
        # Evaluate the model
        y_pred_prob = self.classifier.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Print detailed evaluation
        print(f"Model accuracy: {accuracy:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))
        
        # Save the model and label encoder
        self.save_model()
        
        return {
            'history': history.history,
            'accuracy': accuracy,
            'report': report,
            'classes': self.label_encoder.classes_
        }
    
    def save_model(self):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.classifier_path), exist_ok=True)
        
        # Save classifier
        self.classifier.save(self.classifier_path)
        print(f"Saved classifier to {self.classifier_path}")
        
        # Save label encoder
        with open(self.label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Saved label encoder to {self.label_encoder_path}")
    
    def recognize(self, face_img, threshold=0.7, unknown_threshold=0.5, min_confidence_margin=0.2):
        if self.classifier is None or self.label_encoder is None:
            raise ValueError("Classifier not loaded. Train or load a classifier first.")
        
        # Generate embedding
        embedding = self.get_embedding(face_img)
        
        # Predict
        embedding = np.expand_dims(embedding, axis=0)
        predictions = self.classifier.predict(embedding)[0]
        
        # Get highest confidence prediction
        max_confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        # Get second highest confidence for margin calculation
        sorted_indices = np.argsort(predictions)[::-1]
        second_best_confidence = predictions[sorted_indices[1]] if len(sorted_indices) > 1 else 0
        confidence_margin = max_confidence - second_best_confidence
        
        #  decision logic for unknown faces
        recognized = False
        name = "Unknown"
    
        if (max_confidence >= threshold and 
            max_confidence >= unknown_threshold and 
            confidence_margin >= min_confidence_margin):
            name = self.label_encoder.classes_[predicted_class]
            recognized = True
        
        return {
            'name': name,
            'confidence': float(max_confidence),
            'confidence_margin': float(confidence_margin),
            'recognized': recognized
        }