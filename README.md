# **Case Management System - Machine Learning and Chatbot**

This project involves building a **machine learning model** to predict case outcomes and an **AI-powered chatbot** to provide insights into the case management system. The project includes data preprocessing, model training, evaluation, and deployment of a chatbot interface.

---

## **Project Overview**

### **Objective**
The goal of this project is to:
1. Build a machine learning model to predict case outcomes based on case management data.
2. Develop an AI-powered chatbot to provide insights into the case management system.
3. This approach helps bridge the gap between technical and non-technical users and in a company setting would make collaboration and data extraction easier, creating impact in both product and operations.

---

### **Key Features**
1. **Machine Learning Model**:
   - Predicts case outcomes (e.g., resolved, pending, escalated).
   - Uses a Random Forest classifier for prediction.
   - Evaluates model performance using accuracy, precision, recall, and F1-score.

2. **Chatbot**:
   - Answers key business questions (e.g., most common case types, factors affecting resolution time).
   - Provides predictions for case outcomes.
   - Uses a smaller model (e.g., DistilGPT-2) for natural language processing.

3. **Dataset**:
   - A synthetic dataset was generated using **Python Faker** to mimic real-world case management data.
   - The dataset includes three tables: `cases`, `clients`, and `assignees`.

---

## **Steps and Process**

### **1. Data Preprocessing**
- The dataset was preprocessed to handle missing values, encode categorical variables, and scale numerical features.
- Categorical variables were one-hot encoded, and numerical features were standardized.

### **2. Model Training**
- A **Random Forest classifier** was trained to predict case outcomes.
- The model was evaluated using accuracy, precision, recall, and F1-score.
- The trained model was saved as `model.pkl` using `joblib`.

### **3. Chatbot Development**
- The chatbot was developed using **Gradio** for the user interface.
- A smaller model (**tiny llama**) was used for natural language processing to ensure faster responses.
- The chatbot can answer questions about case trends, resolution times, and assignee performance.

### **4. Dataset Generation**
- A synthetic dataset was generated using **Python Faker** to create realistic case management data.
- The dataset includes tables for cases, clients, and assignees, with attributes such as case type, priority, risk level, and resolution time.

---
## Tools Used for research
1. Google
2 .Chatgpt
---

## **How to Run the Project**

### **1. Install Dependencies**
Install the required libraries:
```bash
pip install -r requirements.txt
```

### **2. Run the Chatbot**
Run the chatbot interface:
```bash
python app.py
```

### **3. Interact with the Chatbot**
Open the Gradio interface in your browser and ask questions like:
- "What are the most common case types?"
- "Predict the outcome for this case."


## **Author**
Joyce Nhlengetwa

