"""
Simple Digit Recognizer App with Pickle Support
A clean Streamlit interface for digit recognition with fast model loading
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

# Initialize session state
if 'recognizer' not in st.session_state:
    st.session_state['recognizer'] = None
if 'models_loaded' not in st.session_state:
    st.session_state['models_loaded'] = False

# Title
st.title("🔢 Simple Digit Recognizer")
st.markdown("A clean, simple interface for recognizing handwritten digits (0-9) with fast model loading!")

# Sidebar
st.sidebar.header("Settings")

# Check if models exist
models_exist = os.path.exists('models') and os.path.exists('models/mlp_model.pkl') and os.path.exists('models/cnn_model.pkl')

if models_exist:
    st.sidebar.success("✅ Pre-trained models found!")
    st.sidebar.markdown("**Models ready for instant use**")
else:
    st.sidebar.warning("⚠️ No pre-trained models found")
    st.sidebar.markdown("**Need to train models first**")

# Training settings (only show if no models exist or force retrain)
with st.sidebar.expander("🔧 Training Settings", expanded=not models_exist):
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    # Training samples selector
    n_samples = st.selectbox(
        "Training Samples",
        [10000, 30000, 50000, 70000],
        index=1,
        help="More samples = better accuracy but slower training"
    )
    
    force_retrain = st.checkbox("Force Retrain Models", value=False, 
                               help="Check this to retrain even if models exist")

st.sidebar.markdown("**Model Architecture:**")
st.sidebar.markdown("- **MLP:** Multi-Layer Perceptron (128, 64 neurons)")
st.sidebar.markdown("- **CNN:** Convolutional Neural Network")
st.sidebar.markdown("- **Dataset:** MNIST")

st.sidebar.markdown("**Prediction Model:**")
pred_model = st.sidebar.radio("Choose model for prediction:", ["MLP", "CNN"], index=1)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Quick Load Button (if models exist)
    if models_exist and not st.session_state['models_loaded']:
        if st.button("⚡ Quick Load Models", type="primary"):
            with st.spinner("Loading pre-trained models..."):
                try:
                    recognizer = SimpleDigitRecognizer(test_size=test_size, n_samples=n_samples)
                    models_loaded = recognizer.load_models()
                    
                    if models_loaded:
                        st.session_state['recognizer'] = recognizer
                        st.session_state['models_loaded'] = True
                        st.success(f"🎉 Models loaded successfully! ({', '.join(models_loaded)})")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load models")
                        
                except Exception as e:
                    st.error(f"Loading failed: {str(e)}")
    
    # Training Button
    train_button_text = "🔄 Retrain Models" if models_exist else "🚀 Train Models"
    if st.button(train_button_text, type="secondary" if models_exist else "primary"):
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
            progress_bar.progress(85)
            
            # Save models
            status_text.text("Saving models as pickle files...")
            recognizer.save_models()
            progress_bar.progress(95)
            
            # Evaluate models
            status_text.text("Evaluating models...")
            results = recognizer.evaluate_models()
            progress_bar.progress(100)
            
            st.success("🎉 Training completed and models saved!")
            st.session_state['recognizer'] = recognizer
            st.session_state['results'] = results
            st.session_state['models_loaded'] = True
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.stop()

with col2:
    if st.session_state['models_loaded'] and st.session_state['recognizer']:
        recognizer = st.session_state['recognizer']
        
        # Show quick model info
        st.header("🤖 Model Status")
        st.success("Models Ready!")
        
        available_models = list(recognizer.models.keys())
        for model in available_models:
            st.write(f"✅ {model} loaded")
        
        # Quick evaluation if data is available
        if hasattr(recognizer, 'X_test') and recognizer.X_test is not None:
            if st.button("📊 Evaluate Models", type="secondary"):
                with st.spinner("Evaluating..."):
                    results = recognizer.evaluate_models()
                    st.session_state['results'] = results

# Display detailed results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    recognizer = st.session_state['recognizer']
    
    st.header("📊 Model Performance")
    
    # Metrics display
    col1, col2 = st.columns(2)
    with col1:
        if 'MLP' in results:
            st.metric(label="🧠 MLP Accuracy", value=f"{results['MLP']['accuracy']:.4f}")
    with col2:
        if 'CNN' in results:
            st.metric(label="🔥 CNN Accuracy", value=f"{results['CNN']['accuracy']:.4f}")
    
    # Detailed metrics table
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
        if st.button("🔄 Show Random Samples"):
            fig = recognizer.plot_sample_images()
            if fig:
                st.pyplot(fig)
                plt.close()
    
    with col2:
        st.subheader("Class Distribution")
        if st.button("📊 Show Distribution"):
            fig = recognizer.plot_class_distribution()
            if fig:
                st.pyplot(fig)
                plt.close()

# Separator
st.markdown("---")

# --- Drawing Canvas Section ---
st.header("🎨 Draw a Digit and Predict!")

# Only show prediction interface if models are loaded
if not st.session_state['models_loaded']:
    st.warning("⚠️ Please load or train models first to use the prediction feature.")
    st.info("💡 Click 'Quick Load Models' if you have pre-trained models, or 'Train Models' to create new ones.")
else:
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
                    st.warning(f"❌ {pred_model} model is not available.")
        else:
            st.info("👆 Draw a digit on the canvas to get started!")

# Instructions
