import pandas as pd
import numpy as np
import os

def generate_hr_data(num_records=2000):
    print("⚙️ Generating Virtual HR Dataset...")
    np.random.seed(42)
    
    # Demographics & Role Context
    departments = np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], num_records)
    job_level = np.random.choice(['Junior', 'Mid', 'Senior'], num_records, p=[0.5, 0.3, 0.2])
    experience_years = np.random.randint(0, 20, num_records)
    
    # Work Signals & Engagement
    training_hours = np.random.randint(0, 100, num_records)
    on_time_delivery_rate = np.random.uniform(0.5, 1.0, num_records)
    manager_score = np.random.uniform(1.0, 5.0, num_records)
    bugs_or_errors = np.random.randint(0, 50, num_records)
    
    # Calculate Target: perf_band_next (H/M/L)
    # Logic: High performance correlates with high delivery, high training, low bugs
    score = (on_time_delivery_rate * 40) + (manager_score * 10) + (training_hours * 0.2) - (bugs_or_errors * 0.5)
    
    conditions = [
        (score >= 70),
        (score >= 40) & (score < 70),
        (score < 40)
    ]
    choices = ['High', 'Medium', 'Low']
    perf_band_next = np.select(conditions, choices, default='Medium')
    
    # Create DataFrame
    df = pd.DataFrame({
        'department': departments,
        'job_level': job_level,
        'experience_years': experience_years,
        'training_hours': training_hours,
        'on_time_delivery_rate': np.round(on_time_delivery_rate, 2),
        'manager_score': np.round(manager_score, 1),
        'bugs_or_errors': bugs_or_errors,
        'perf_band_next': perf_band_next
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/employee_data.csv', index=False)
    print("✅ Dataset saved to data/employee_data.csv")

if __name__ == "__main__":
    generate_hr_data()