# 📈 Employee Performance Predictor using Data Analytics

## 📌 Project Overview
An industry-oriented People Analytics platform that uses historical HR telemetry (training hours, delivery rates, manager scores) to forecast upcoming employee performance bands. This helps HR teams identify high performers, reduce appraisal bias, and proactively target L&D (Learning & Development) interventions.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Random Forest, ColumnTransformer, Pipelines)
* **Data Processing:** Pandas, NumPy
* **Frontend UI:** Streamlit

## 📐 System Architecture
1. **Data Simulator:** Generates synthetic, bias-free HR logs avoiding sensitive attributes.
2. **Preprocessing Pipeline:** Applies median imputation and robust scaling for continuous variables, and One-Hot Encoding for categorical context.
3. **Classification Engine:** A class-balanced Random Forest classifier predicting performance bands (High, Medium, Low).
4. **Actionability Dashboard:** A Streamlit UI converting predictions into human-readable L&D nudges.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Generate virtual data: `python src/generate_data.py`
3. Train the ML model: `python src/train_model.py`
4. Launch Dashboard: `streamlit run app.py`

## 📊 Results & Evaluation
* Optimized for **macro-F1 score** to accurately balance predictions across all performance tiers.
* Provides real-time actionable recommendations for manager coaching and training workshops based on predicted risk.