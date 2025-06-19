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
        self.models_trained = False

    def load_data(self):
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        X = X.astype(np.float32) / 255.0
        if self.n_samples < len(X):
            X, y = X[:self.n_samples], y[:self.n_samples]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(f"Dataset loaded: {len(self.X_train)} training, {len(self.X_test)} testing samples")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess_data(self):
        print("Preprocessing data...")
        self.X_train_mlp = self.scaler.fit_transform(self.X_train)
        self.X_test_mlp = self.scaler.transform(self.X_test)
        print("Data normalized")
        return self.X_train_mlp, self.X_test_mlp

    def check_models_exist(self, directory='models'):
        """Check if pre-trained models exist"""
        mlp_path = os.path.join(directory, "mlp_model.pkl")
        cnn_path = os.path.join(directory, "cnn_model.pkl")
        scaler_path = os.path.join(directory, "scaler.pkl")
        data_info_path = os.path.join(directory, "model_info.pkl")
        
        return (os.path.exists(mlp_path) and 
                os.path.exists(cnn_path) and 
                os.path.exists(scaler_path) and 
                os.path.exists(data_info_path))

    def train_mlp(self, max_iter=100):
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
        print("Training CNN...")
        X_train_cnn = self.X_train.reshape(-1, 28, 28, 1)
        X_test_cnn = self.X_test.reshape(-1, 28, 28, 1)
        y_train_cat = keras.utils.to_categorical(self.y_train, 10)
        y_test_cat = keras.utils.to_categorical(self.y_test, 10)
        
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
        img = image_array.reshape(28, 28)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Center the digit
        cy, cx = ndimage.center_of_mass(img)
        if not (np.isnan(cx) or np.isnan(cy)):
            shift_x = 14 - cx
            shift_y = 14 - cy
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (28, 28))
        
        # Normalize
        if img.max() > 0:
            img = img / img.max()
        
        # Dilate to make lines thicker
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = np.clip(img, 0, 1)
        
        return img.flatten()

    def predict_digit(self, image_array):
        predictions = {}
        processed_img = self.preprocess_canvas_image(image_array)
        
        if 'MLP' in self.models:
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
        """Save all models and metadata as pickle files"""
        os.makedirs(directory, exist_ok=True)
        
        # Save MLP model
        if 'MLP' in self.models:
            with open(f"{directory}/mlp_model.pkl", 'wb') as f:
                pickle.dump(self.models['MLP'], f)
            print("✅ MLP model saved as pickle")
        
        # Save CNN model (convert to pickle-friendly format)
        if 'CNN' in self.models:
            # Save CNN weights and architecture separately for better pickle support
            cnn_data = {
                'weights': self.models['CNN'].get_weights(),
                'config': self.models['CNN'].get_config()
            }
            with open(f"{directory}/cnn_model.pkl", 'wb') as f:
                pickle.dump(cnn_data, f)
            print("✅ CNN model saved as pickle")
        
        # Save scaler
        with open(f"{directory}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ Scaler saved as pickle")
        
        # Save model metadata
        model_info = {
            'n_samples': self.n_samples,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'models_available': list(self.models.keys())
        }
        with open(f"{directory}/model_info.pkl", 'wb') as f:
            pickle.dump(model_info, f)
        print("✅ Model info saved as pickle")
        
        print(f"🎉 All models saved to {directory}/ as pickle files!")

    def load_models(self, directory='models'):
        """Load all models from pickle files"""
        models_loaded = []
        
        try:
            # Check if models directory exists
            if not os.path.exists(directory):
                print(f"❌ Models directory '{directory}' not found")
                return models_loaded
            
            # Load model info
            info_path = os.path.join(directory, "model_info.pkl")
            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    model_info = pickle.load(f)
                print(f"📊 Found models trained with {model_info['n_samples']} samples")
            
            # Load scaler
            scaler_path = os.path.join(directory, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Scaler loaded from pickle")
            
            # Load MLP model
            mlp_path = os.path.join(directory, "mlp_model.pkl")
            if os.path.exists(mlp_path):
                with open(mlp_path, 'rb') as f:
                    self.models['MLP'] = pickle.load(f)
                models_loaded.append('MLP')
                print("✅ MLP model loaded from pickle")
            
            # Load CNN model
            cnn_path = os.path.join(directory, "cnn_model.pkl")
            if os.path.exists(cnn_path):
                with open(cnn_path, 'rb') as f:
                    cnn_data = pickle.load(f)
                
                # Reconstruct CNN model
                model = keras.Sequential.from_config(cnn_data['config'])
                model.set_weights(cnn_data['weights'])
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                self.models['CNN'] = model
                models_loaded.append('CNN')
                print("✅ CNN model loaded from pickle")
            
            if models_loaded:
                self.models_trained = True
                print(f"🎉 Successfully loaded models: {models_loaded}")
            else:
                print("❌ No saved models found")
                
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            models_loaded = []
        
        return models_loaded

    def quick_load_or_train(self, force_retrain=False):
        """Quick method to load existing models or train new ones"""
        if not force_retrain and self.check_models_exist():
            print("🚀 Found existing models, loading...")
            models_loaded = self.load_models()
            if models_loaded:
                print("⚡ Models loaded successfully! Ready for predictions.")
                return True
        
        print("🔄 Training new models...")
        self.load_data()
        self.preprocess_data()
        self.train_mlp()
        self.train_cnn()
        self.save_models()
        self.models_trained = True
        return True

    def plot_sample_images(self, n_samples=8):
        if self.X_train is None:
            print("❌ No training data loaded")
            return None
            
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
        if self.y_train is None or self.y_test is None:
            print("❌ No training/testing data loaded")
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set distribution
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        sns.barplot(x=train_counts.index, y=train_counts.values, ax=ax1)
        ax1.set_title('Training Set Distribution')
        ax1.set_xlabel('Digit')
        ax1.set_ylabel('Count')
        
        # Testing set distribution
        test_counts = pd.Series(self.y_test).value_counts().sort_index()
        sns.barplot(x=test_counts.index, y=test_counts.values, ax=ax2)
        ax2.set_title('Testing Set Distribution')
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        return fig

def main():
    print("🔢 Simple Digit Recognizer with Pickle Support")
    print("=" * 50)
    
    recognizer = SimpleDigitRecognizer(n_samples=30000)
    
    # Quick load or train
    recognizer.quick_load_or_train()
    
    # Evaluate models if needed
    if recognizer.X_test is None:
        recognizer.load_data()
        recognizer.preprocess_data()
    
    results = recognizer.evaluate_models()
    
    print("\n🎉 Models ready for use!")
    print("You can now use the Streamlit app for digit recognition.")

if __name__ == "__main__":
    main()