# -----------------------------------
# تدريب النموذج
# -----------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_model():
    # تحميل البيانات
    df = pd.read_csv("Training.csv")

    # تنظيف البيانات
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols:
        df[col] = df[col].replace(0, np.nan)

    df.fillna(df.mean(), inplace=True)

    # تقسيم البيانات
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # تدريب النموذج
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, X.columns