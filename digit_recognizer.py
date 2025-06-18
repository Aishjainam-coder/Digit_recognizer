"""
Simple Digit Recognizer
A clean, simple implementation for recognizing handwritten digits (0-9)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import plotly.express as px
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
from scipy import ndimage

class SimpleDigitRecognizer:
    """Simple digit recognizer with improved preprocessing and multiple models"""
    
    def __init__(self, test_size=0.2, random_state=42, n_samples=10000):
        self.test_size = test_size
        self.random_state = random_state
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load MNIST dataset (optionally use only a subset for speed)"""
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Normalize to 0-1 range
        X = X.astype(np.float32) / 255.0
        
        # Use only a subset for faster training
        if self.n_samples < len(X):
            X, y = X[:self.n_samples], y[:self.n_samples]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Dataset loaded: {len(self.X_train)} training, {len(self.X_test)} testing samples")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_data(self):
        """Normalize the data for MLP"""
        print("Preprocessing data...")
        # For MLP, we need standardization
        self.X_train_mlp = self.scaler.fit_transform(self.X_train)
        self.X_test_mlp = self.scaler.transform(self.X_test)
        print("Data normalized")
        return self.X_train_mlp, self.X_test_mlp
    
    def train_mlp(self, max_iter=100):
        """Train MLP model with better settings"""
        print("Training MLP...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=max_iter,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=10,
            verbose=False,
            learning_rate_init=0.001,
            batch_size=256,
            alpha=0.0001
        )
        mlp.fit(self.X_train_mlp, self.y_train)
        self.models['MLP'] = mlp
        print("MLP trained")
        return mlp
    
    def train_cnn(self, epochs=15, batch_size=128):
        """Train CNN model with improved architecture"""
        print("Training CNN...")
        
        # Reshape for CNN: (samples, 28, 28, 1)
        X_train_cnn = self.X_train.reshape(-1, 28, 28, 1)
        X_test_cnn = self.X_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(self.y_train, 10)
        y_test_cat = keras.utils.to_categorical(self.y_test, 10)
        
        # Improved CNN architecture
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        ]
        
        model.fit(
            X_train_cnn, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(X_test_cnn, y_test_cat),
            callbacks=callbacks
        )
        
        self.models['CNN'] = model
        print("CNN trained")
        return model
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            if name == 'MLP':
                y_pred = model.predict(self.X_test_mlp)
            elif name == 'CNN':
                X_test_cnn = self.X_test.reshape(-1, 28, 28, 1)
                y_pred = model.predict(X_test_cnn)
                y_pred = y_pred.argmax(axis=1)
            else:
                continue
                
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            print(f"{name}: {accuracy:.4f} accuracy")
            
        return results
    
    def preprocess_canvas_image(self, image_array):
        """Improved preprocessing for canvas images to match MNIST style"""
        # Reshape to 28x28
        img = image_array.reshape(28, 28)
        
        # Apply Gaussian blur to smooth the image
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Center the digit in the image using center of mass
        cy, cx = ndimage.center_of_mass(img)
        
        if not (np.isnan(cx) or np.isnan(cy)):
            # Calculate shift needed to center the digit
            shift_x = 14 - cx
            shift_y = 14 - cy
            
            # Apply the shift
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (28, 28))
        
        # Normalize to match MNIST preprocessing
        if img.max() > 0:
            img = img / img.max()
        
        # Apply slight dilation to make lines thicker (like MNIST)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        
        # Final normalization
        img = np.clip(img, 0, 1)
        
        return img.flatten()
    
    def predict_digit(self, image_array):
        """Predict digit with improved preprocessing"""
        predictions = {}
        
        # Preprocess the image
        processed_img = self.preprocess_canvas_image(image_array)
        
        # MLP prediction
        if 'MLP' in self.models:
            # Standardize for MLP
            arr = processed_img.reshape(1, -1)
            arr = self.scaler.transform(arr)
            pred = self.models['MLP'].predict(arr)[0]
            
            try:
                prob = self.models['MLP'].predict_proba(arr)[0]
                confidence = np.max(prob)
            except:
                confidence = 1.0
                
            predictions['MLP'] = {
                'prediction': int(pred),
                'confidence': float(confidence)
            }
        
        # CNN prediction
        if 'CNN' in self.models:
            arr = processed_img.reshape(1, 28, 28, 1)
            prob = self.models['CNN'].predict(arr, verbose=0)[0]
            pred = np.argmax(prob)
            confidence = np.max(prob)
            
            predictions['CNN'] = {
                'prediction': int(pred),
                'confidence': float(confidence)
            }
        
        return predictions
    
    def save_models(self, directory='models'):
        """Save trained models"""
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'CNN':
                # Save Keras model
                model.save(f"{directory}/cnn_model.h5")
            else:
                # Save sklearn model
                filename = f"{directory}/{name.lower().replace(' ', '_')}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
        
        # Save scaler
        with open(f"{directory}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"Models saved to {directory}/")
    
    def plot_sample_images(self, n_samples=8):
        """Plot sample images from dataset"""
        indices = np.random.choice(len(self.X_train), n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            img = self.X_train[idx].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Digit: {self.y_train[idx]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_class_distribution(self):
        """Plot class distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        sns.barplot(x=train_counts.index, y=train_counts.values, ax=ax1)
        ax1.set_title('Training Set Distribution')
        ax1.set_xlabel('Digit')
        ax1.set_ylabel('Count')
        
        # Testing set
        test_counts = pd.Series(self.y_test).value_counts().sort_index()
        sns.barplot(x=test_counts.index, y=test_counts.values, ax=ax2)
        ax2.set_title('Testing Set Distribution')
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        return fig

def main():
    """Main function to run the digit recognizer"""
    print("🔢 Simple Digit Recognizer")
    print("=" * 40)
    
    # Initialize recognizer (use more samples for better accuracy)
    recognizer = SimpleDigitRecognizer(n_samples=30000)
    
    # Load and preprocess data
    recognizer.load_data()
    recognizer.preprocess_data()
    
    # Train models
    recognizer.train_mlp()
    recognizer.train_cnn()
    
    # Evaluate models
    results = recognizer.evaluate_models()
    
    # Save models
    recognizer.save_models()
    
    print("\n🎉 Training completed!")
    print("Models saved to 'models/' directory")

if __name__ == "__main__":
    main()