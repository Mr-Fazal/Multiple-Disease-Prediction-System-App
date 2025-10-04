<!-- TITLE & BADGES -->
# 🧠 Multiple Disease Prediction System App  

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-ff4b4b?logo=streamlit)](https://streamlit.io/)  
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-brightgreen?logo=scikitlearn)](https://scikit-learn.org/)  
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Integrated-orange?logo=tensorflow)](https://tensorflow.org/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)  
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red)](#)  

---

## ✨ Overview  

The **Multiple Disease Prediction System App** 🧬 is an intelligent web application built using **Machine Learning** and **Deep Learning** in **Python**.  
This app can **predict the likelihood of multiple diseases** — including **Diabetes 🩸**, **Heart Disease ❤️**, and **Parkinson’s Disease 🧠** — based on user-inputted medical parameters.  

Developed in **PyCharm** and deployed using **Streamlit**, the project combines **data science, healthcare, and AI** to provide fast, accurate, and user-friendly predictions.  

💡 **Highlight:**  
A one-stop predictive health assistant that helps users identify early warning signs for major diseases using smart algorithms and interactive web visualization. 🚀  

---

## 🩺 Supported Disease Models  

| Disease | Model Type | Technique Used | Accuracy |
|----------|-------------|----------------|-----------|
| 🩸 Diabetes | Machine Learning | Support Vector Machine (SVM) | ~97% |
| ❤️ Heart Disease | Machine Learning | Logistic Regression | ~94% |
| 🧠 Parkinson’s Disease | Deep Learning | Neural Network (TensorFlow/Keras) | ~95% |

---

## 🧱 Tech Stack  

| Category | Technology Used |
|-----------|-----------------|
| 💻 Programming Language | Python |
| 🧠 ML/DL Libraries | Scikit-learn, TensorFlow, Keras |
| 🧮 Data Handling | Pandas, NumPy |
| 📊 Visualization | Matplotlib, Seaborn |
| 🌐 Web Framework | Streamlit |
| 🧰 IDE | PyCharm |

---

## 📊 Dataset Information  

All datasets are obtained from **Kaggle**:  
- **Diabetes Dataset:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Heart Disease Dataset:** [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)  
- **Parkinson’s Dataset:** [Kaggle - Parkinson’s Disease Dataset](https://www.kaggle.com/datasets/debasisdotcom/parkinsons-disease-dataset)  

Each dataset includes multiple medical parameters (like glucose level, blood pressure, BMI, etc.) used for model training and evaluation.  

---

## 🔍 Project Flow  

## 🔍 Project Flow  

```mermaid
graph TD
A[Import Dataset] --> B[Data Cleaning and Preprocessing]
B --> C[Exploratory Data Analysis (EDA)]
C --> D[Model Training with ML and DL Algorithms]
D --> E[Model Evaluation and Accuracy Testing]
E --> F[Deploy Model using Streamlit]
F --> G[Predict Diseases based on User Inputs]


🚀 Features

✅ Predicts Diabetes, Heart Disease, and Parkinson’s Disease with high accuracy.
✅ Interactive Streamlit web interface for real-time predictions.
✅ Multiple input fields for medical parameters.
✅ ML & DL hybrid approach for robust predictions.
✅ Visualization dashboard for insights and model results.
✅ Responsive design using HTML, CSS, and Bootstrap-inspired layout.
✅ User-friendly deployment directly via Streamlit or localhost.

Installation & Setup
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/Multiple-Disease-Prediction-System-App.git

# 2️⃣ Navigate into the project directory
cd Multiple-Disease-Prediction-System-App

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Streamlit app
streamlit run app.py

🧠 Model Training Steps

1️⃣ Import Dataset — from Kaggle datasets for all three diseases.
2️⃣ Preprocess Data — clean missing values, normalize data, split for training/testing.
3️⃣ Train Models — using SVM, Logistic Regression, and Neural Networks.
4️⃣ Evaluate Models — accuracy, confusion matrix, precision-recall.
5️⃣ Integrate Models — combine predictions into a unified app interface.
6️⃣ Deploy via Streamlit — user inputs → predictions → displayed results.

📸 Screenshots
App Homepage
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cd90ec68-0f44-4476-ad6f-2a4ea8d1955e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9c566111-67d2-4d3d-8b57-fb4fa397e6f3" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/18ebb12d-d4b5-49cf-b6eb-1b4cb29f0781" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3e0ffc38-76dc-4dda-a7f0-c7ca3a3bf029" />

📈 Example Output
🔹 Entered Parameters:
Glucose: 140 | Blood Pressure: 80 | BMI: 28.5 | Age: 45

✅ Prediction: The patient is likely *not diabetic*.

💡 Future Enhancements

🔹 Integrate more diseases like liver or kidney disease.

🔹 Add user authentication & history tracking.

🔹 Deploy on cloud platforms (AWS, Heroku).

🔹 Enable real-time data input via wearable devices.

📜 License

This project is licensed under the MIT License — free to use and modify with proper attribution.

⭐ If you found this project helpful, don’t forget to star the repo!
🩺 Stay healthy, stay informed — powered by AI. 🤖
