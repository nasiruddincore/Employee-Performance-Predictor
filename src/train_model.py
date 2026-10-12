import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate():
    print("🧠 Loading data and engineering features...")
    df = pd.read_csv("data/employee_data.csv")
    
    # Split Features and Target
    target = 'perf_band_next'
    y = df[target]
    X = df.drop(columns=[target])
    
    # Identify column types
    cat_cols = X.select_dtypes(include='object').columns
    num_cols = X.select_dtypes(include='number').columns
    
    # Preprocessing Pipeline
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    
    # Complete Pipeline with Balanced Random Forest
    model_pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=13, n_estimators=200))
    ])
    
    # Stratified Split to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, stratify=y)
    
    print("🏋️ Training Random Forest Model...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluation
    print("\n📊 Evaluation Metrics:")
    y_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, "models/employee_perf_model.pkl")
    print("\n✅ Model saved to models/employee_perf_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()