# 🫀 ECG Normal vs Abnormal Classification

A deep learning-based web application for classifying **ECG (Electrocardiogram) images** as **Normal** or **Abnormal**.

The project uses **TensorFlow/Keras and MobileNetV2** for image classification, with a **Flask backend** serving the trained model and a web-based frontend for uploading ECG images and displaying predictions.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice or clinical evaluation.

---

## 🚀 Live Demo

🌐 **Application:**
https://cardioscan.onrender.com/

---

## 📌 Project Overview

Electrocardiograms (ECGs) contain important information about the electrical activity of the heart. Interpreting ECG recordings requires medical expertise and can be time-consuming.

This project explores the application of **deep learning and transfer learning** for automatic ECG image classification.

The system accepts an ECG image and processes it through a trained deep learning model to classify the image as either:

* 🟢 **Normal**
* 🔴 **Abnormal**

The application provides the prediction along with a confidence score through a simple web interface.

---

## ✨ Features

* 🫀 ECG image classification
* 🤖 Deep learning-based prediction
* 🧠 MobileNetV2 architecture
* 🔄 Transfer learning
* 🌐 Interactive web interface
* 📤 ECG image upload
* ⚡ Flask REST API
* 📊 Prediction confidence score
* 📱 Browser-based interface
* ☁️ Deployment on Render
* 🔌 CORS-enabled backend

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    └──────────┬───────────┘
                               │
                               │ Upload ECG Image
                               ▼
                    ┌──────────────────────┐
                    │    Web Frontend      │
                    │     index.html       │
                    └──────────┬───────────┘
                               │
                               │ POST /predict
                               ▼
                    ┌──────────────────────┐
                    │    Flask Backend     │
                    │       run.py         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │                      │
                    │ Resize → 224 × 224   │
                    │ RGB Conversion       │
                    │ Normalization        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   TensorFlow Model   │
                    │     MobileNetV2      │
                    │                      │
                    │    ecg_model.h5      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Prediction       │
                    │                      │
                    │ Normal / Abnormal    │
                    │ Confidence Score     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    JSON Response     │
                    └──────────────────────┘
```

---

## 🧠 Machine Learning Model

The project uses **MobileNetV2**, a lightweight convolutional neural network architecture, for ECG image classification.

MobileNetV2 is well suited for image classification applications because it provides efficient feature extraction while maintaining a relatively lightweight architecture.

The trained model is stored in:

```text
ecg_model.h5
```

The model receives a preprocessed ECG image and produces a probability used to determine the final classification.

### Classification Logic

The prediction uses a probability threshold of `0.5`.

```python
prob = model.predict(img_array)[0][0]

label = "Normal" if prob < 0.5 else "Abnormal"
```

Therefore:

```text
Probability < 0.5
        ↓
     Normal

Probability ≥ 0.5
        ↓
    Abnormal
```

---

## 🔄 Image Preprocessing

Before an ECG image is passed to the model, it goes through the following preprocessing pipeline:

```text
Original ECG Image
        │
        ▼
Resize to 224 × 224
        │
        ▼
Convert to RGB
        │
        ▼
Convert to NumPy Array
        │
        ▼
Normalize Pixel Values
        │
        ▼
Add Batch Dimension
        │
        ▼
MobileNetV2 Model
        │
        ▼
Prediction
```

The preprocessing includes:

```python
IMG_SIZE = (224, 224)

img = image.resize(IMG_SIZE).convert("RGB")
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
```

---

## 🛠️ Technology Stack

### Machine Learning

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow
* Scikit-learn

### Backend

* Flask
* Flask-CORS
* Gunicorn

### Frontend

* HTML
* CSS
* JavaScript

### Deployment

* Render

---

## 📂 Project Structure

```text
normal-abnormal-ecg/
│
├── ecg_model.h5          # Trained TensorFlow/Keras model
│
├── run.py                # Flask backend and prediction API
│
├── index.html            # Web frontend
│
├── requirements.txt      # Python dependencies
│
└── README.md             # Project documentation
```

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/satyam-tomar/normal-abnormal-ecg.git
```

---

## 2. Navigate to the Project Directory

```bash
cd normal-abnormal-ecg
```

---

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Flask server:

```bash
python run.py
```

The application will be available locally at:

```text
http://localhost:5002
```

Open the URL in your browser to access the ECG classification interface.

---

# 🔌 API Documentation

## `POST /predict`

The `/predict` endpoint accepts an ECG image and returns the model's classification result.

### Request

```text
POST /predict
Content-Type: multipart/form-data
```

The uploaded image should be provided using the:

```text
file
```

field.

### Example Using cURL

```bash
curl -X POST \
  -F "file=@path/to/ecg_image.jpg" \
  http://localhost:5002/predict
```

### Example Response

```json
{
    "label": "Normal",
    "confidence": "92.4%",
    "accuracy": "92.5%",
    "precision": "90.0%",
    "recall": "93.0%",
    "f1_score": "91.5%",
    "r2_score": "0.88"
}
```

> **Note:** The `label` and `confidence` are generated from model inference. The additional evaluation values currently returned by the API are static values in the existing implementation rather than being calculated for each individual uploaded image.

---

# 📊 Evaluation Metrics

The project includes common machine learning evaluation metrics:

| Metric    | Description                                                      |
| --------- | ---------------------------------------------------------------- |
| Accuracy  | Measures the overall percentage of correct predictions           |
| Precision | Measures how many predicted positive cases are actually positive |
| Recall    | Measures how many actual positive cases are correctly identified |
| F1 Score  | Harmonic mean of precision and recall                            |
| R² Score  | Statistical measure used to represent explained variance         |

The project includes an evaluation function using **Scikit-learn** for calculating classification-related metrics.

> For a production-quality medical ML system, these metrics should be evaluated on a properly separated and representative test dataset.

---

# 🔬 How the Application Works

## Step 1 — Upload ECG Image

The user selects an ECG image through the web interface.

```text
User
 ↓
Upload ECG Image
```

---

## Step 2 — Send Image to Backend

The frontend sends the uploaded image to the Flask API:

```text
POST /predict
```

---

## Step 3 — Image Preprocessing

The backend processes the image:

```text
Resize
   ↓
RGB Conversion
   ↓
NumPy Array
   ↓
Normalization
   ↓
Batch Dimension
```

---

## Step 4 — Model Inference

The processed image is passed into the trained TensorFlow model.

```text
ECG Image
    ↓
Preprocessing
    ↓
MobileNetV2
    ↓
Probability
```

---

## Step 5 — Classification

The model probability is compared against the classification threshold.

```text
             Model Probability
                    │
          ┌─────────┴─────────┐
          │                   │
       < 0.5                ≥ 0.5
          │                   │
          ▼                   ▼
       NORMAL              ABNORMAL
```

---

## Step 6 — Display Result

The Flask backend returns a JSON response containing the prediction and confidence information.

The frontend displays the result to the user.

---

# 🎯 Use Cases

This project can be used as an educational and research demonstration for:

* Medical image classification
* Deep learning
* Transfer learning
* Computer vision
* Healthcare AI
* TensorFlow/Keras
* Flask API development
* Machine learning deployment
* Image preprocessing
* Model inference

---

# 🔮 Future Improvements

The system can be extended with several improvements:

* [ ] Add a complete model training pipeline
* [ ] Use a dedicated validation and test dataset
* [ ] Add confusion matrix visualization
* [ ] Add ROC-AUC evaluation
* [ ] Add class-wise performance metrics
* [ ] Add Grad-CAM visualizations
* [ ] Add explainable AI
* [ ] Support multiple ECG abnormality classes
* [ ] Improve model calibration
* [ ] Add model versioning
* [ ] Add automated API testing
* [ ] Add Docker support
* [ ] Improve API validation and error handling
* [ ] Add file type and image size validation
* [ ] Add authentication
* [ ] Add API rate limiting
* [ ] Add model monitoring
* [ ] Improve deployment infrastructure

---

# ⚠️ Limitations

This project is primarily an **educational and research-oriented machine learning application**.

It should not be considered a clinical diagnostic system.

Some important limitations include:

* ECG image quality can vary considerably.
* Different ECG machines may produce different image formats.
* Model performance depends heavily on the quality and diversity of the training dataset.
* Performance on unseen populations and ECG formats requires separate validation.
* A binary classification model cannot capture every possible ECG abnormality.
* Predictions should be interpreted carefully.
* The current API returns predefined evaluation metrics rather than calculating them for every prediction.

**Do not use this application for medical diagnosis or treatment decisions. Always consult a qualified healthcare professional for medical interpretation.**

---

# 📚 Technical Background

Deep learning and transfer learning have been widely explored for ECG classification and healthcare-related image analysis.

Convolutional Neural Networks (CNNs) can learn visual patterns from ECG representations, while transfer-learning architectures such as **MobileNetV2** can provide efficient feature extraction.

This project demonstrates how a trained computer vision model can be integrated into a complete application consisting of:

```text
Machine Learning Model
        +
Image Processing
        +
Flask REST API
        +
Web Interface
        +
Cloud Deployment
```

---

# ☁️ Deployment

The application is deployed using **Render**.

### Production Application

```text
https://cardioscan.onrender.com/
```

The deployment architecture can be represented as:

```text
User Browser
     │
     ▼
Render Deployment
     │
     ▼
Flask Application
     │
     ▼
TensorFlow Model
     │
     ▼
Prediction
```

---

# 🤝 Contributing

Contributions and improvements are welcome.

### 1. Fork the Repository

### 2. Create a Feature Branch

```bash
git checkout -b feature/new-feature
```

### 3. Make Your Changes

### 4. Commit Changes

```bash
git commit -m "Add new feature"
```

### 5. Push the Branch

```bash
git push origin feature/new-feature
```

### 6. Create a Pull Request

---

# 👨‍💻 Contributors

This project was collaboratively developed by:

### Satyam Tomar

GitHub:
https://github.com/satyam-tomar

### Uday Kumar

GitHub:
https://github.com/Udayy08

---

# 🤝 Collaboration

This project was developed collaboratively with contributions across **machine learning, computer vision, backend API development, frontend integration, and deployment**.

The project combines a trained deep learning model with a Flask-based inference API and a web interface to create an end-to-end ECG image classification application.

---

# ⭐ Support

If you found this project useful for learning, research, or experimentation, consider giving the repository a ⭐ on GitHub.

---

# 🫀 Built With

```text
Python
TensorFlow
Keras
MobileNetV2
Flask
NumPy
Pillow
Scikit-learn
HTML
CSS
JavaScript
Render
```

---

## 📄 License

Please refer to the repository for the applicable licensing terms before redistributing or using this project commercially.
