# Neural Network Predicting Chances of Admission at UCLA 
 
# 🎓 UCLA Admission Predictor

This Streamlit application predicts the probability of admission to UCLA based on a user's academic profile. It leverages a trained neural network model to give personalized predictions and provides interactive visuals to understand each factor's influence.

---

## 📌 Overview

This project includes:
- A trained MLP (Multi-layer Perceptron) model
- User-friendly form inputs for scores and credentials
- Interactive visualizations (gauge charts, bar plots)
- Feature importance breakdown per prediction

---

## 🛠 Project Structure

```
ucla_admission_predictor/
├── app.py                       # Streamlit application
├── models/
│   └── admission_nn_model.pkl  # Trained MLP model
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 🔧 Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ucla_admission_predictor.git
cd ucla_admission_predictor
```

### 🌐 Step 2: Set Up a Virtual Environment

```bash
python -m venv venv
# Activate the environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

Your app will launch in a browser window at [http://localhost:8501](http://localhost:8501).

---

## 🎯 Features

- **Academic Input Form**: Enter GRE, TOEFL, CGPA, SOP, LOR, and Research.
- **Gauge Charts**: View score levels visually.
- **Admission Prediction**: See your chance of admission in real time.
- **Feature Breakdown**: Visualize how each input contributes to the prediction.
- **Responsive UI**: Sidebar controls for toggling graphs and metrics.

---

## 🧠 Model

- **Model Type**: Multi-layer Perceptron (Neural Network)
- **Framework**: Trained using `scikit-learn` and saved with `joblib`
- **Input Features**:
  - GRE Score
  - TOEFL Score
  - SOP, LOR
  - CGPA
  - University Rating (encoded)
  - Research Experience (binary)

---

## 📈 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Joblib
- Plotly
- Pandas & NumPy

---

## 🌐 Deployment

Deploy to [Streamlit Cloud](https://streamlit.io/cloud) easily:
- Push this app to a GitHub repository
- Deploy via Streamlit Cloud
- Set `app.py` as the main entry point

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 🙌 Contribution

Feel free to fork the repo, open issues, or submit pull requests for improvements or new features.

---

## ✉️ Contact

- **Author:** [Your Name Here]  
- **GitHub:** [https://github.com/just-sree](https://github.com/just-sree)

