# -----------------------------------
# تطبيق Flask
# -----------------------------------

from flask import Flask, render_template, request
import pandas as pd
from model import train_model

# إنشاء التطبيق
app = Flask(__name__)

# تحميل النموذج
model, columns = train_model()

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template('index.html', columns=columns)

# عند الضغط على زر التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # قراءة القيم من النموذج
        input_data = [float(request.form[col]) for col in columns]

        # تحويلها إلى DataFrame
        input_df = pd.DataFrame([input_data], columns=columns)

        # التنبؤ
        probability = model.predict_proba(input_df)[0][1]

        result = f"{probability * 100:.2f}%"

        return render_template('index.html', columns=columns, result=result)

    except:
        return render_template('index.html', columns=columns, result="خطأ في الإدخال")

# تشغيل التطبيق
#if __name__ == "__main__":
    app.run(debug=True)
import os
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)