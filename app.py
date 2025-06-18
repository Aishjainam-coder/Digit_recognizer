"""
Simple Digit Recognizer App
A clean Streamlit interface for digit recognition
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image, ImageOps, ImageFilter
import pickle
import os
from streamlit_drawable_canvas import st_canvas
import cv2

from digit_recognizer import SimpleDigitRecognizer

# Page config
st.set_page_config(
    page_title="Simple Digit Recognizer",
    page_icon="🔢",
    layout="wide"
)

# Title
st.title("🔢 Simple Digit Recognizer")
st.markdown("A clean, simple interface for recognizing handwritten digits (0-9)")

# Sidebar
st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
st.sidebar.markdown("**Model:** Multi-Layer Perceptron (MLP) + CNN")
st.sidebar.markdown("**Dataset:** MNIST")

# Training samples selector
n_samples = st.sidebar.selectbox(
    "Training Samples",
    [10000, 30000, 50000, 70000],
    index=1,
    help="More samples = better accuracy but slower training"
)

st.sidebar.markdown("**Prediction Model:**")
pred_model = st.sidebar.radio("Choose model for prediction:", ["MLP", "CNN"], index=1)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Initializing..."):
            recognizer = SimpleDigitRecognizer(test_size=test_size, n_samples=n_samples)
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load data
            status_text.text("Loading MNIST dataset...")
            recognizer.load_data()
            progress_bar.progress(20)
            
            # Preprocess data
            status_text.text("Preprocessing data...")
            recognizer.preprocess_data()
            progress_bar.progress(35)
            
            # Train MLP
            status_text.text("Training MLP (Neural Network)...")
            recognizer.train_mlp(max_iter=100)
            progress_bar.progress(60)
            
            # Train CNN
            status_text.text("Training CNN (Convolutional Neural Network)...")
            recognizer.train_cnn(epochs=15, batch_size=128)
            progress_bar.progress(90)
            
            # Evaluate models
            status_text.text("Evaluating models...")
            results = recognizer.evaluate_models()
            progress_bar.progress(100)
            
            st.success("🎉 Training completed!")
            st.session_state['recognizer'] = recognizer
            st.session_state['results'] = results
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.stop()

with col2:
    if 'results' in st.session_state:
        results = st.session_state['results']
        st.header("📊 Results")
        
        if 'MLP' in results:
            st.metric(label="MLP Accuracy", value=f"{results['MLP']['accuracy']:.4f}")
        if 'CNN' in results:
            st.metric(label="CNN Accuracy", value=f"{results['CNN']['accuracy']:.4f}")

# Display detailed results if training is done
if 'results' in st.session_state:
    results = st.session_state['results']
    recognizer = st.session_state['recognizer']
    
    # Metrics table
    st.subheader("Detailed Metrics")
    metrics_data = []
    for name in ["MLP", "CNN"]:
        if name in results:
            y_pred = results[name]['predictions']
            from sklearn.metrics import classification_report
            report = classification_report(recognizer.y_test, y_pred, output_dict=True, zero_division=0)
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{results[name]['accuracy']:.4f}",
                'Precision': f"{report['macro avg']['precision']:.4f}",
                'Recall': f"{report['macro avg']['recall']:.4f}",
                'F1-Score': f"{report['macro avg']['f1-score']:.4f}"
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    # Data visualization
    st.header("📈 Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sample Images")
        fig = recognizer.plot_sample_images()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Class Distribution")
        fig = recognizer.plot_class_distribution()
        st.pyplot(fig)
        plt.close()
    
    # Save model
    st.header("💾 Save Models")
    if st.button("Save Trained Models"):
        recognizer.save_models()
        st.success("Models saved to 'models/' directory!")

# --- Drawing Canvas Section ---
st.header("🎨 Draw a Digit and Predict!")
st.markdown("""
Draw a digit (0-9) in the canvas below and click **Predict** to see what the model thinks! 

**Tips for better predictions:**
- Draw thick, clear digits
- Center your digit in the canvas
- Use the full height/width of the canvas
- Make sure your digit is connected (no broken lines)
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Draw Here:")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",  # Transparent fill
        stroke_width=20,  # Thicker stroke
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )
    st.caption("💡 Use thick strokes and center your digit for best results!")

with col2:
    st.subheader("Preview & Predict")
    
    if canvas_result.image_data is not None:
        # Show original drawing
        st.image(canvas_result.image_data, width=140, caption="Your Drawing")
        
        # Improved preprocessing
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        
        # Invert colors (white background, black digit -> black background, white digit)
        img = ImageOps.invert(img)
        
        # Apply slight blur to smooth the drawing
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Resize to 28x28 with good resampling
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Show processed image
        processed_img_display = (img_array * 255).astype(np.uint8)
        st.image(processed_img_display, width=140, caption="Processed (28x28)")
        
        if st.button("🔮 Predict", key="predict_canvas", type="primary"):
            if 'recognizer' not in st.session_state:
                st.warning("⚠️ Please train the models first using 'Start Training' above.")
            else:
                recognizer = st.session_state['recognizer']
                
                with st.spinner("Predicting..."):
                    predictions = recognizer.predict_digit(img_array)
                
                st.subheader("🎯 Prediction Results:")
                
                if pred_model in predictions:
                    pred_info = predictions[pred_model]
                    
                    # Display prediction with confidence
                    confidence = pred_info['confidence']
                    prediction = pred_info['prediction']
                    
                    # Color code confidence
                    if confidence > 0.8:
                        confidence_color = "🟢"
                    elif confidence > 0.5:
                        confidence_color = "🟡"
                    else:
                        confidence_color = "🔴"
                    
                    st.markdown(f"""
                    **{pred_model} Prediction:** 
                    
                    # {prediction}
                    
                    {confidence_color} **Confidence:** {confidence:.3f}
                    """)
                    
                    # Show all predictions for comparison
                    if len(predictions) > 1:
                        st.subheader("🔍 All Model Predictions:")
                        comparison_data = []
                        for model_name, pred_data in predictions.items():
                            comparison_data.append({
                                'Model': model_name,
                                'Prediction': pred_data['prediction'],
                                'Confidence': f"{pred_data['confidence']:.3f}"
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                    
                    # Confidence visualization
                    if confidence > 0.3:  # Only show if confidence is reasonable
                        fig = px.bar(
                            x=[pred_model],
                            y=[confidence],
                            title=f"Model Confidence Level",
                            labels={'x': 'Model', 'y': 'Confidence'},
                            color=[confidence],
                            color_continuous_scale='RdYlGn',
                            range_color=[0, 1]
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Low confidence prediction. Try redrawing the digit more clearly.")
                        
                else:
                    st.warning(f"❌ {pred_model} model is not trained yet.")
    else:
        st.info("👆 Draw a digit on the canvas to get started!")

# Instructions
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. **Start Training**: Click to train both MLP and CNN models
2. **View Results**: See model performance and accuracy metrics
3. **Draw Digits**: Use the canvas to draw and predict digits
4. **Choose Model**: Select MLP or CNN for predictions

**Drawing Tips:**
- Use thick, bold strokes
- Center your digit
- Make digits large and clear
- Ensure lines are connected
""")

# About
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This app uses machine learning to recognize handwritten digits:

- **MLP**: Multi-Layer Perceptron with 2 hidden layers
- **CNN**: Convolutional Neural Network with batch normalization
- **Dataset**: MNIST (handwritten digits)
- **Preprocessing**: Image centering, normalization, and smoothing

**Improvements made:**
- Better image preprocessing
- Improved CNN architecture
- Enhanced training parameters
- Better digit centering algorithm
""")

# Performance tips
with st.sidebar.expander("🚀 Performance Tips"):
    st.markdown("""
    **For better accuracy:**
    - Use 30,000+ training samples
    - Train for more epochs (CNN)
    - Draw digits clearly and boldly
    - Center digits in canvas
    
    **For faster training:**
    - Use 10,000 samples
    - Reduce epochs to 5-10
    - Use smaller batch sizes
    """)