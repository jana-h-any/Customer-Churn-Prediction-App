
##### 📊 Customer Churn Prediction


## 📌 Overview

This project predicts customer churn for a telecom company based on customer demographics, service usage, and account information.
It uses **machine learning classification models** to identify customers who are most likely to leave, allowing proactive retention actions.

The workflow includes:

1. **Data Preprocessing** – Handling missing values, encoding categorical variables, scaling numerical features, and balancing the dataset with SMOTE.
2. **Model Training** – Testing multiple classification algorithms.
3. **Model Selection** – Choosing the best-performing model based on **Recall** for the churn class.
4. **Deployment** – Creating an interactive **Streamlit app** to make real-time churn predictions.

---

## ⚙️ Models Tested & Recall Scores

| Model                 | Recall (Churn=1) |
| --------------------- | ---------------- |
| Logistic Regression   | 0.84             |
| SVC                   | 0.80             |
| KNN                   | 0.77             |
| Decision Tree (Tuned) | 0.93             |

**Chosen Model:** **Decision Tree (Tuned)** – achieved the highest recall (0.93) after hyperparameter tuning with **GridSearchCV**.

---

## 🚀 How It Works

1. **User Input** – The app takes customer details (gender, tenure, contract type, monthly charges, etc.).
2. **Preprocessing Pipeline** – Applies SMOTE, scaling, and encoding as in training.
3. **Prediction** – Model outputs the probability of churn.
4. **Threshold** – Customers with probability ≥ 0.3 are classified as “Will Churn”.

---

## 🛠 Tech Stack

* **Python**
* **Pandas, NumPy**
* **scikit-learn**
* **imbalanced-learn (SMOTE)**
* **Streamlit** (for the web app)
* **Joblib** (for model saving/loading)

---

## 📂 Project Structure

```
📁 churn_prediction
│── churn_noteboook.ipynb   # Full training workflow
│── pipeline_model.pkl      # Saved tuned Decision Tree model
│── app.py                  # Streamlit app
│── README.md               # Project documentation
```

---

## 🎯 Model Deployment (Streamlit App)

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📺 Demo

Try the live app here: [Churn Prediction App](https://customer-churn-prediction-app-mkx4dmmniyqrtbjqkrq9de.streamlit.app/)

---

##👩‍💻 Developed By

Jana Hany Mostafa

##📬 Contact

GitHub: @jana-h-any

LinkedIn: [linkedin.com/in/jana-hany]

Email: [janahanymostafa016@gmail.com]
