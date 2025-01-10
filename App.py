import streamlit as st
import joblib
import pandas as pd

# تحميل النماذج المحفوظة
log_model = joblib.load("logistic_regression_model.pkl")
import zipfile
import os

# تحديد مسار الملف المضغوط
zip_file = 'random_forest_model.zip'
extract_folder = 'model_folder'  # المسار الذي سيتم فك الضغط فيه

# فك الضغط
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# الآن لديك الملف داخل المجلد 'model_folder'
model_path = os.path.join(extract_folder, 'random_forest_model.pkl')

rf_model = joblib.load("random_forest_model.pkl")

# عنوان التطبيق
st.title("تنبؤ مرض الزهايمر")

# إنشاء مدخلات المستخدم
age = st.slider("العمر", 0, 100, 50)
memory_loss_severity = st.slider("شدة فقدان الذاكرة", 1, 10, 5)
cognitive_impairment_score = st.slider("درجة العجز المعرفي", 1, 10, 5)
family_history = st.selectbox("تاريخ عائلي للمرض", ["نعم", "لا"])
lifestyle_score = st.slider("درجة نمط الحياة", 1, 10, 5)
education_level = st.selectbox("المستوى التعليمي", ["ابتدائي", "ثانوي", "جامعي", "دراسات عُليا"])
physical_activity_hours = st.slider("عدد ساعات النشاط البدني أسبوعيًا", 0, 40, 10)
chronic_illness_count = st.slider("عدد الأمراض المزمنة", 0, 5, 1)
smoking_status = st.selectbox("حالة التدخين", ["مدخن", "غير مدخن"])

# تحويل المدخلات النصية إلى قيم رقمية
family_history = 1 if family_history == "نعم" else 0
education_level = {"ابتدائي": 0, "ثانوي": 1, "جامعي": 2, "دراسات عُليا": 3}[education_level]
smoking_status = {"مدخن": 1, "غير مدخن": 0}[smoking_status]

# إنشاء DataFrame للمدخلات
input_data = pd.DataFrame({
    "Age": [age],
    "MemoryLossSeverity": [memory_loss_severity],
    "CognitiveImpairmentScore": [cognitive_impairment_score],
    "FamilyHistory": [family_history],
    "LifestyleScore": [lifestyle_score],
    "EducationLevel": [education_level],
    "PhysicalActivityHours": [physical_activity_hours],
    "ChronicIllnessCount": [chronic_illness_count],
    "SmokingStatus": [smoking_status]
})

# استخدام النموذج للتنبؤ
log_prediction = log_model.predict(input_data)
rf_prediction = rf_model.predict(input_data)

# عرض النتائج
st.write(f"نتيجة التنبؤ باستخدام Logistic Regression: {'إيجابي' if log_prediction[0] == 1 else 'سلبي'}")
st.write(f"نتيجة التنبؤ باستخدام Random Forest: {'إيجابي' if rf_prediction[0] == 1 else 'سلبي'}")
