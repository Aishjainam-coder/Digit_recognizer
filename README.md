# 🔢 Simple Digit Recognizer

A clean, simple machine learning project for recognizing handwritten digits (0-9).

## 🚀 Features

- **Simple & Clean**: Everything in just 2 main files
- **Multiple Models**: SVM, Random Forest, MLP (Neural Network)
- **Interactive App**: Beautiful Streamlit interface
- **Easy to Use**: Just run and see results
- **Image Upload**: Test your own handwritten digits

## 📁 Project Structure

```
digit_recognizer/
├── digit_recognizer.py    # Main ML logic (single file)
├── app.py                # Streamlit web app
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 How to Run

### Option 1: Interactive Web App (Recommended)

```bash
streamlit run app.py
```

This opens a beautiful web interface where you can:
- Configure training settings
- Watch real-time training progress
- See model performance comparisons
- Upload and test your own images

### Option 2: Command Line

```bash
python digit_recognizer.py
```

This runs the training directly and saves models to a `models/` folder.

## 📊 What You'll See

### In the Web App:

1. **Settings Panel** (Sidebar):
   - Adjust test size
   - Select which models to train

2. **Training Process**:
   - Real-time progress bars
   - Live status updates

3. **Results**:
   - Model accuracy comparison charts
   - Detailed metrics table
   - Sample digit images
   - Class distribution plots

4. **Interactive Testing**:
   - Upload your own digit images
   - Get predictions from all models
   - See confidence levels

## 📈 Expected Results

With default settings, you should see:
- **SVM**: ~95-97% accuracy
- **Random Forest**: ~96-98% accuracy  
- **MLP**: ~97-99% accuracy

## 🎓 What You'll Learn

This project demonstrates:
- **Data Loading**: MNIST dataset handling
- **Preprocessing**: Data normalization
- **Model Training**: Multiple ML algorithms
- **Evaluation**: Performance metrics
- **Visualization**: Data and results analysis
- **Interactive ML**: Real-time predictions

## 📚 Technologies Used

- **Python**: Core language
- **Scikit-learn**: Machine learning
- **Streamlit**: Web interface
- **Plotly**: Interactive charts
- **Matplotlib/Seaborn**: Static plots
- **NumPy/Pandas**: Data handling

## 🚀 Quick Start

1. Install: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. Open browser and click "Start Training"
4. Upload images to test!

---

**Simple, Clean, Effective! 🎉** 