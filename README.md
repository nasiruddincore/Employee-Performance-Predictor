# 📈 Employee Performance Predictor using Data Analytics

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Frontend](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

An industry-oriented People Analytics platform that uses historical HR telemetry (training hours, delivery rates, manager scores) to forecast upcoming employee performance bands. This helps HR teams identify high performers early, reduce appraisal bias, and proactively target Learning & Development (L&D) interventions.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [System Objectives](#-system-objectives)
- [Industry Relevance & Use Cases](#-industry-relevance--use-cases)
- [System Architecture & Data Flow](#-system-architecture--data-flow)
- [Folder Structure](#-folder-structure)
- [Installation & Setup Guide](#-installation--setup-guide)
- [How to Run (Execution Workflow)](#-how-to-run-execution-workflow)
- [Model Evaluation Results](#-model-evaluation-results)
- [Author & License](#-author--license)

---

## 📌 Project Overview
Instead of waiting for a yearly review to discover an employee is struggling, this system evaluates daily work habits and engagement metrics to alert managers months in advance. The AI acts as a highly observant HR assistant, classifying future performance into High, Medium, or Low bands and generating targeted coaching recommendations.

---

## ⚠️ Problem Statement
* **Reactive Appraisals:** Traditional yearly reviews address poor performance after the damage is done, leading to costly turnover and backfill hiring.
* **Human Bias:** Promotion and rating calibrations are often influenced by recency bias or subjective manager opinions rather than objective data.
* **Inefficient L&D Spending:** Training budgets are frequently wasted on generic courses instead of targeted upskilling for employees who actually need it.

---

## 🎯 System Objectives
1. **Virtual HR Data Generation:** Simulate a realistic, bias-free dataset containing 2,000+ employee records, avoiding sensitive attributes while maintaining statistical correlations.
2. **Predictive Modeling:** Train a class-balanced Random Forest classifier to predict multi-class performance bands.
3. **Actionability Engine:** Map feature shortfalls (e.g., low on-time delivery) to concrete HR interventions (e.g., sprint planning workshops).
4. **Interactive Dashboard:** Provide a Streamlit UI for managers to input metrics and instantly receive AI-driven performance forecasts.

---

## 🏭 Industry Relevance & Use Cases
* **Enterprise HR (Google, IBM, TCS):** Used to identify future leaders for fast-track promotion and leadership programs.
* **Manager Coaching:** Highlights employees who need Performance Improvement Plan (PIP) support before the appraisal cycle ends.
* **Workforce Retention:** Catches disengagement early to prevent expensive employee attrition.

---

## 📐 System Architecture & Data Flow

```text
[ Virtual HR Data Generator ] ──(Synthetic Logs)──► [ data/employee_data.csv ]
                                                            │
                                                  (Feature Engineering)
                                                            ▼
[ Streamlit HR Dashboard ] ◄──────(Inference)───── [ Random Forest ML Model ]

🚀 How to Run (Execution Workflow)

Step 1: Generate HR Dataset

Run the simulator to create synthetic employee demographic and productivity data.
Bash

python src/generate_data.py

Step 2: Train the ML Model

Extract features, encode categorical data, scale numerical data, and train the Random Forest Classifier.
Bash

python src/train_model.py

Step 3: Launch Web Dashboard Simulation

Start the interactive UI for real-time manager input and prediction generation.
Bash

streamlit run app.py

📊 Model Evaluation Results

    Macro-F1 Score: Optimized to ensure accurate predictions across all imbalanced classes (High/Medium/Low).

    Pipeline Integrity: Utilizes ColumnTransformer with median imputation and robust scaling to prevent data leakage.

    Explainability: Surfaces key operational drivers (e.g., training hours, bug counts) alongside predictions to ensure ethical interventions.

👤 Author & License

Developed as an industry-oriented Machine Learning & Data Analytics portfolio project. Distributed under the MIT License.
