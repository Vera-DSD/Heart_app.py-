import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path

# ================== НАСТРОЙКА СТРАНИЦЫ ==================
st.set_page_config(
    page_title="🫀 Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ================== ЗАГРУЗКА МОДЕЛИ ==================
@st.cache_resource
def load_model_and_features():
    """Загружаем модель и информацию о фичах"""
    try:
        # Загрузка модели
        model_path = Path("best_model.pkl")
        if not model_path.exists():
            st.error("❌ Файл модели не найден: best_model.pkl")
            st.info("Убедитесь, что модель сохранена в той же папке")
            return None, None
        
        model = joblib.load(model_path)
        
        # Правильные названия фичей (из ваших данных)
        feature_names = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 
            'MaxHR', 'Oldpeak', 'Sex', 'ChestPainType', 
            'RestingECG', 'ExerciseAngina', 'ST_Slope'
        ]
        
        st.sidebar.success("✅ Модель успешно загружена")
        return model, feature_names
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {str(e)}")
        return None, None

# Загружаем модель
model, FEATURE_NAMES = load_model_and_features()

# ================== ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ ==================
st.title("🫀 Heart Disease Prediction App")
st.markdown("""
Это приложение использует модель машинного обучения для предсказания 
риска сердечных заболеваний на основе медицинских показателей.
""")

# Разделяем на две колонки
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Демографические данные")
    
    # Основные параметры
    age = st.slider("Возраст (Age)", 20, 100, 50, 
                   help="Возраст пациента в годах")
    
    sex = st.selectbox("Пол (Sex)", 
                      ["Мужской", "Женский"],
                      help="Биологический пол пациента")
    
    cp = st.selectbox("Тип боли в груди (ChestPainType)", 
                     ["ASY", "ATA", "NAP", "TA"],
                     help="ASY: Бессимптомный, ATA: Атипичная стенокардия, NAP: Неангинальная боль, TA: Типичная стенокардия")
    
    trestbps = st.slider("Артериальное давление в покое (RestingBP)", 
                        80, 200, 120,
                        help="Артериальное давление в мм рт.ст.")

with col2:
    st.subheader("💊 Биохимические показатели")
    
    chol = st.slider("Уровень холестерина (Cholesterol)", 
                    100, 600, 200,
                    help="Уровень холестерина в мг/дл")
    
    fbs = st.selectbox("Уровень сахара натощак > 120 мг/дл (FastingBS)", 
                      [0, 1],
                      format_func=lambda x: "Да" if x == 1 else "Нет",
                      help="Показатель уровня глюкозы")
    
    thalach = st.slider("Максимальный пульс (MaxHR)", 
                       60, 220, 150,
                       help="Максимальная достигнутая частота сердечных сокращений")
    
    oldpeak = st.slider("Депрессия ST (Oldpeak)", 
                       0.0, 6.0, 1.0, 0.1,
                       help="Депрессия ST, вызванная физической нагрузкой относительно покоя")

# Дополнительные параметры
with st.expander("📈 Дополнительные параметры ЭКГ"):
    col3, col4 = st.columns(2)
    
    with col3:
        restecg = st.selectbox("Результат ЭКГ в покое (RestingECG)", 
                              ["Normal", "LVH", "ST"],
                              help="Normal: Норма, LVH: Гипертрофия левого желудочка, ST: Аномалии ST-T")
    
    with col4:
        exang = st.selectbox("Стенокардия при нагрузке (ExerciseAngina)", 
                           [0, 1],
                           format_func=lambda x: "Да" if x == 1 else "Нет",
                           help="Наличие стенокардии, вызванной физической нагрузкой")
    
    slope = st.selectbox("Наклон сегмента ST (ST_Slope)", 
                        ["Up", "Flat", "Down"],
                        help="Наклон пикового сегмента ST при нагрузке")

# ================== КОДИРОВАНИЕ ДАННЫХ ==================
def encode_features(age_val, sex_val, cp_val, trestbps_val, chol_val, 
                   fbs_val, thalach_val, oldpeak_val, restecg_val, 
                   exang_val, slope_val):
    """Кодируем категориальные признаки в числовые"""
    
    # Кодирование пола
    sex_encoded = 1 if sex_val == "Мужской" else 0
    
    # Кодирование типа боли в груди
    cp_mapping = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
    cp_encoded = cp_mapping.get(cp_val, 0)
    
    # Кодирование результатов ЭКГ
    restecg_mapping = {"Normal": 0, "LVH": 1, "ST": 2}
    restecg_encoded = restecg_mapping.get(restecg_val, 0)
    
    # Кодирование наклона ST
    slope_mapping = {"Up": 0, "Flat": 1, "Down": 2}
    slope_encoded = slope_mapping.get(slope_val, 0)
    
    # Создаем словарь с ВСЕМИ признаками в правильном порядке
    encoded_data = {
        'Age': float(age_val),
        'RestingBP': float(trestbps_val),
        'Cholesterol': float(chol_val),
        'FastingBS': float(fbs_val),
        'MaxHR': float(thalach_val),
        'Oldpeak': float(oldpeak_val),
        'Sex': float(sex_encoded),
        'ChestPainType': float(cp_encoded),
        'RestingECG': float(restecg_encoded),
        'ExerciseAngina': float(exang_val),
        'ST_Slope': float(slope_encoded)
    }
    
    return encoded_data

# ================== ПРЕДСКАЗАНИЕ ==================
st.markdown("---")
predict_col1, predict_col2 = st.columns([1, 3])

with predict_col1:
    predict_btn = st.button("🎯 Сделать предсказание", 
                          type="primary",
                          use_container_width=True)

if predict_btn and model is not None:
    try:
        # Кодируем данные
        encoded_data = encode_features(
            age, sex, cp, trestbps, chol, fbs, 
            thalach, oldpeak, restecg, exang, slope
        )
        
        # СОЗДАЕМ DataFrame с ТОЧНЫМ порядком признаков
        input_df = pd.DataFrame([encoded_data])
        
        # Убеждаемся, что порядок столбцов правильный
        input_df = input_df[FEATURE_NAMES]
        
        # Проверяем типы данных
        input_df = input_df.astype(float)
        
        # Делаем предсказание
        with st.spinner("🧠 Анализируем данные..."):
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
        
        # Отображаем результат
        st.markdown("---")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.error(f"## 🚨 Высокий риск сердечного заболевания")
                st.metric(label="Вероятность", 
                         value=f"{probabilities[1]*100:.1f}%",
                         delta="Требуется консультация врача",
                         delta_color="inverse")
            else:
                st.success(f"## ✅ Низкий риск сердечного заболевания")
                st.metric(label="Вероятность", 
                         value=f"{probabilities[0]*100:.1f}%",
                         delta="Рекомендуется профилактика")
        
        with result_col2:
            # Визуализация вероятностей
            prob_df = pd.DataFrame({
                'Состояние': ['Низкий риск', 'Высокий риск'],
                'Вероятность': [probabilities[0], probabilities[1]]
            })
            
            st.bar_chart(prob_df.set_index('Состояние'))
        
        # Детальная информация
        with st.expander("📋 Детали предсказания"):
            st.write("**Введенные данные:**")
            st.dataframe(input_df.T.rename(columns={0: 'Значение'}))
            
            st.write("**Распределение вероятностей:**")
            prob_details = pd.DataFrame({
                'Класс': ['Низкий риск (0)', 'Высокий риск (1)'],
                'Вероятность': [f"{probabilities[0]*100:.2f}%", 
                               f"{probabilities[1]*100:.2f}%"]
            })
            st.table(prob_details)
            
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {str(e)}")
        st.write("**Отладочная информация:**")
        st.write(f"Тип данных: {type(input_df)}")
        st.write(f"Размер данных: {input_df.shape}")
        st.write(f"Колонки: {list(input_df.columns)}")

elif predict_btn:
    st.warning("⚠️ Модель не загружена. Убедитесь, что файл best_model.pkl находится в папке.")

# ================== САЙДБАР С ИНФОРМАЦИЕЙ ==================
with st.sidebar:
    st.header("ℹ️ Информация")
    
    st.markdown("""
    ### О модели
    - **Алгоритм:** CatBoost
    - **Признаки:** 11 медицинских показателей
    - **Точность:** ≈88%
    
    ### Как использовать
    1. Заполните все поля формы
    2. Нажмите кнопку "Сделать предсказание"
    3. Оцените результат и рекомендации
    
    ### Интерпретация признаков
    - **Age:** Возраст пациента
    - **RestingBP:** Давление в покое
    - **Cholesterol:** Уровень холестерина
    - **FastingBS:** Сахар натощак
    - **MaxHR:** Максимальный пульс
    - **Oldpeak:** Депрессия ST
    """)
    
    # Кнопка для демо-данных
    if st.button("📋 Загрузить демо-данные", use_container_width=True):
        st.session_state.demo_loaded = True
        st.rerun()
    
    if 'demo_loaded' in st.session_state and st.session_state.demo_loaded:
        st.info("Демо-данные загружены. Заполните форму для тестирования.")

# ================== ФУТЕР ==================
st.markdown("---")
st.caption("""
⚠️ **Важно:** Это приложение предназначено только для образовательных целей. 
Для медицинской диагностики обратитесь к квалифицированному специалисту.
""")

# ================== ФУНКЦИЯ ДЛЯ ТЕСТИРОВАНИЯ ==================
def test_prediction():
    """Функция для тестирования предсказания"""
    test_data = {
        'Age': 52.0,
        'RestingBP': 125.0,
        'Cholesterol': 212.0,
        'FastingBS': 0.0,
        'MaxHR': 168.0,
        'Oldpeak': 1.0,
        'Sex': 1.0,
        'ChestPainType': 2.0,
        'RestingECG': 0.0,
        'ExerciseAngina': 0.0,
        'ST_Slope': 1.0
    }
    
    test_df = pd.DataFrame([test_data])[FEATURE_NAMES]
    return model.predict(test_df)[0] if model else None

# Автоматический тест при загрузке
if model is not None and 'test_done' not in st.session_state:
    try:
        test_result = test_prediction()
        st.session_state.test_done = True
    except:
        pass