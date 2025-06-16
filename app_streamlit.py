import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data():
    # Load dataset (ganti dengan path dataset Anda)
    df = pd.read_csv('rumah.csv')
    
    # Create price categories
    bins = [0, df['price_in_rp'].quantile(0.33), df['price_in_rp'].quantile(0.66), df['price_in_rp'].max()]
    labels = ['Murah', 'Sedang', 'Mahal']
    df['kategori_harga'] = pd.cut(df['price_in_rp'], bins=bins, labels=labels, include_lowest=True)
    
    # Select relevant features
    features = ['bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2', 
                'floors', 'building_age', 'garages', 'property_type', 
                'furnishing', 'property_condition']
    
    df = df[features + ['kategori_harga']]
    
    # Fill missing values
    df.fillna({
        'building_age': df['building_age'].median(),
        'garages': 0,
        'property_condition': 'unknown',
        'furnishing': 'unknown'
    }, inplace=True)
    
    # Encode categorical features
    encode_map = {
        'property_type': {'rumah': 1, 'unknown': 0},
        'furnishing': {'unfurnished': 3, 'furnished': 0, 'semi furnished': 2, 'unknown': 1},
        'property_condition': {'bagus': 0, 'bagus sekali': 1, 'unknown': 2}
    }
    
    for col, mapping in encode_map.items():
        df[col] = df[col].map(mapping)
    
    return df, encode_map

df, encode_map = load_and_preprocess_data()

# 2. Train and Save Model
@st.cache_resource
def train_and_save_model():
    X = df.drop('kategori_harga', axis=1)
    y = df['kategori_harga']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Save model with metadata
    model_data = {
        'model': model,
        'encode_map': encode_map,
        'features': X.columns.tolist(),
        'classes': model.classes_
    }
    
    joblib.dump(model_data, 'model_rumah.sav')
    
    return model, X_test, y_test

try:
    model, X_test, y_test = train_and_save_model()
except Exception as e:
    st.error(f"Error in model training: {str(e)}")
    st.stop()

# 3. Streamlit App
st.title("Klasifikasi Harga Rumah 🏠")
st.write("Masukkan informasi properti untuk memprediksi kategori harga rumah")

with st.form("form_rumah"):
    col1, col2 = st.columns(2)
    
    with col1:
        bedrooms = st.number_input("Jumlah Kamar Tidur", 0, 10, 2)
        bathrooms = st.number_input("Jumlah Kamar Mandi", 0, 10, 1)
        land_size_m2 = st.number_input("Luas Tanah (m²)", 0, 1000, 100)
        building_size_m2 = st.number_input("Luas Bangunan (m²)", 0, 1000, 80)
        floors = st.selectbox("Jumlah Lantai", [1, 2, 3])
        
    with col2:
        building_age = st.number_input("Usia Bangunan (tahun)", 0, 100, 5)
        garages = st.selectbox("Jumlah Garasi", [0, 1, 2, 3])
        property_type = st.selectbox("Tipe Properti", ['rumah', 'unknown'])
        furnishing = st.selectbox("Perabotan", ['unfurnished', 'furnished', 'semi furnished', 'unknown'])
        property_condition = st.selectbox("Kondisi Properti", ['bagus', 'bagus sekali', 'unknown'])
    
    submit = st.form_submit_button("Prediksi")

if submit:
    # Validate inputs
    if any(pd.isna([bedrooms, bathrooms, land_size_m2, building_size_m2])):
        st.error("Harap isi semua field numerik")
        st.stop()
    
    # Prepare input data
    input_data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'land_size_m2': land_size_m2,
        'building_size_m2': building_size_m2,
        'floors': floors,
        'building_age': building_age,
        'garages': garages,
        'property_type': encode_map['property_type'].get(property_type, 0),
        'furnishing': encode_map['furnishing'].get(furnishing, 1),
        'property_condition': encode_map['property_condition'].get(property_condition, 2)
    }
    
    # Convert to DataFrame
    data_input = pd.DataFrame([input_data])[model.feature_names_in_]
    
    # Predict
    try:
        pred = model.predict(data_input)[0]
        st.success(f"Prediksi kategori harga rumah: **{pred}**")
    except Exception as e:
        st.error(f"Error saat prediksi: {str(e)}")
        st.write("Detail input:", input_data)

# 4. Model Evaluation
st.header("Evaluasi Model")
st.write(f"Akurasi Model: {accuracy_score(y_test, model.predict(X_test)):.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_,
            ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Feature Importance
st.subheader("Pentingnya Fitur")
importances = pd.DataFrame({
    'Fitur': model.feature_names_in_,
    'Penting': model.feature_importances_
}).sort_values('Penting', ascending=False)
st.bar_chart(importances.set_index('Fitur'))

# Decision Tree Visualization
st.subheader("Visualisasi Decision Tree (Depth=3)")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model, 
          feature_names=model.feature_names_in_, 
          class_names=model.classes_,
          filled=True, 
          rounded=True,
          max_depth=3,
          ax=ax)
st.pyplot(fig)