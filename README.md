# ❤️ Advanced Cardiology Dashboard

A sophisticated, AI-enhanced medical application designed for predicting cardiovascular risks accurately. This platform uses an intelligent **K-Nearest Neighbors (KNN)** Machine Learning Model in the backend, while operating as a fully featured generative application using **Google's Gemini 2.5 AI** to deliver personalized clinical recommendations.

---

## ✨ Features at a Glance

*   **⚡ Native Machine Learning Inference**: Uses an imported `.pkl` KNN model combined with StandardScaler logic to precisely predict heart disease likelihood instantly.
*   **🧠 Gemini AI Integration**: Deeply integrates with the `google-genai` SDK to supply two major features:
    1.  A fully conversational **Clinical Health Assistant Chatbot** embedded into the interface.
    2.  A **Dynamic Recommendations Engine** that hybridizes your ML risk-score with real-time generative advice on your diet and exercise.
*   **🎨 Premium Glassmorphism UI**: High-end Next.js-inspired aesthetic featuring frosted card containers, gradient drop-shadows, fixed flexbox proportions, and custom typography all achieved through custom Streamlit CSS injection.
*   **📊 Interactive Plotly Gauges**: Transforms raw probability outputs into color-coded safe-zone visualizations.
*   **📄 Enterprise PDF Generation**: Generates downloadable "Hospital Style Diagnostic Reports" using `fpdf`, tagging each report with unique Session UUIDs and timestamping.

---

## 🛠️ Technology Stack

| Component               | Technology                        |
| :---------------------- | :-------------------------------- |
| **Frontend UI**         | Streamlit (with Custom CSS / Streamlit Markdown) |
| **Data Manipulation**   | Pandas, Scikit-Learn              |
| **Data Visualization**  | Plotly (graph_objects)            |
| **AI Generation Engine**| Google GenAI (`gemini-2.5-flash`) |
| **Report Generation**   | FPDF (Python)                     |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Make sure you have python installed natively on your desktop. Then, clone or download this repository locally.

### 2. Install Dependencies
Open a terminal in the root folder of this project and run the following command to retrieve all necessary packages:
```bash
pip install -r requirements.txt
```

*(Note: If you run into environment pathing errors, try `python -m pip install -r requirements.txt`)*

### 3. Execution
Launch the Streamlit dashboard on your localhost port by running:
```bash
streamlit run app.py
```

### 4. Setup Gemini AI
Once the UI loads, verify that your provided Gemini API key configuration is active (it is currently hard-coded intelligently in the source code). Ask the chatbot a test question like *"What are good foods for reducing cholesterol?"* to ensure the Google endpoint is live.

---

## 📂 Project Structure

```text
├── app.py                 # The core application containing UI routes and Prediction flow
├── KNN_heart_model.pkl    # Pre-trained SKLearn K-Nearest-Neighbors classifier
├── scaler.pkl             # Standard Scaler file ensuring input normalization
├── columns.pkl            # Original model column architecture
├── requirements.txt       # Dependencies requirement package
└── README.md              # Documentation
```

---

## ⚕️ Legal & Medical Disclaimer
*This product was developed as a programmatic demonstration of machine learning techniques acting with LLM integrations. It is completely for educational and technical use-cases. Under no circumstances should the outputs, visual bars, diagnostic PDF reports, or chatbot conversations generated within this application be used directly as professional medical advice.*
