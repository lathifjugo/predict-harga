import streamlit as st
import pandas as pd
import joblib
from xhtml2pdf import pisa
import io
from datetime import datetime
import base64

# Load model dan data asli
model_data = joblib.load('model_regresi_rumah.sav')
df = pd.read_csv('rumah.csv')

model = model_data['model']
features = model_data['features']
encode_map = model_data['encode_map']
metrics = model_data.get('metrics', {})

# Streamlit Config
st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="centered")

# --- Custom CSS untuk Background Transparan dengan Gambar Rumah Mewah ---
st.markdown("""
    <style>
    :root {
        --primary: rgba(43, 43, 43, 0.85);
        --secondary: rgba(59, 59, 59, 0.9);
        --text: #ffffff;
        --accent: #4f8bf9;
        --card: rgba(66, 66, 66, 0.9);
    }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url('https://images.unsplash.com/photo-1600585154340-be6161a56a0c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: var(--text);
    }
    
    /* Main container styling */
    .main .block-container {
        background-color: var(--primary);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Result box styling */
    .result-box {
        padding: 25px;
        background-color: var(--card);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        text-align: center;
        border: 1px solid var(--accent);
        margin: 20px 0;
        color: var(--text);
    }
    
    /* Similar property cards */
    .similar-property {
        background-color: var(--secondary);
        backdrop-filter: blur(5px);
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid var(--accent);
        color: var(--text);
    }
    
    .similar-property a {
        color: var(--accent) !important;
        text-decoration: none;
    }
    
    .similar-property a:hover {
        text-decoration: underline;
    }
    
    /* Form elements styling */
    .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, 
    .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as,
    .stNumberInput, .stSelectbox, .stTextInput {
        background-color: var(--secondary) !important;
        backdrop-filter: blur(2px);
    }
    
    .stButton>button {
        background-color: var(--accent) !important;
        color: white !important;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Notification styling */
    .stWarning {
        background-color: #ff6b6b !important;
        color: white !important;
    }
    
    .stSuccess {
        background-color: #51a351 !important;
        color: white !important;
    }
    
    .stError {
        background-color: #bd362f !important;
        color: white !important;
    }
    
    /* Footer styling */
    .footer {
        background-color: rgba(43, 43, 43, 0.8);
        text-align: center;
        font-size: 0.85em;
        color: #aaaaaa;
        padding: 15px;
        border-radius: 10px;
        margin-top: 30px;
    }
    
    /* Divider styling */
    .stDivider>div>div>div {
        background-color: var(--accent) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Fungsi Konversi HTML ke PDF ---
def convert_html_to_pdf(source_html):
    output = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(source_html), dest=output)
    if pisa_status.err:
        return None
    return output

# --- Header ---
st.title("🏠 Prediksi Harga Rumah")
st.markdown("Masukkan detail properti untuk memprediksi harga dalam Rupiah")

# --- Form Input ---
with st.form("form_rumah"):
    st.subheader("📋 Detail Properti")
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Jumlah Kamar Tidur", 0, 10, 2)
        bathrooms = st.number_input("Jumlah Kamar Mandi", 0, 10, 1)
        land_size = st.number_input("Luas Tanah (m²)", 0, 1000, 100)
        building_size = st.number_input("Luas Bangunan (m²)", 0, 1000, 80)
        property_type = st.selectbox("Tipe Properti", ['rumah'])

    with col2:
        floors = st.selectbox("Jumlah Lantai", [1, 2, 3])
        building_age = st.number_input("Usia Bangunan (tahun)", 0, 100, 5)
        garages = st.selectbox("Jumlah Garasi", [0, 1, 2, 3])
        furnishing = st.selectbox("Perabotan", ['unfurnished', 'furnished', 'semi furnished'])
        property_condition = st.selectbox("Kondisi Properti", ['bagus', 'bagus sekali'])

    submit = st.form_submit_button("🔍 Prediksi Harga")

# --- Feature Importance Visualization ---
st.subheader("📊 Pentingnya Fitur dalam Prediksi Harga")

# Dapatkan importance scores dari model
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Fitur': features,
    'Pengaruh': feature_importance
}).sort_values('Pengaruh', ascending=False)

# Tampilkan dalam bentuk bar chart
st.bar_chart(importance_df.set_index('Fitur'))

# Tampilkan penjelasan
st.markdown("""
**Penjelasan Pentingnya Fitur:**
- Nilai di atas menunjukkan seberapa besar pengaruh setiap fitur terhadap prediksi harga rumah
- Semakin tinggi nilainya, semakin besar pengaruh fitur tersebut dalam menentukan harga
""")

# Tampilkan tabel detail
st.write("Detail Pengaruh Setiap Fitur:")
st.dataframe(importance_df.style.format({'Pengaruh': '{:.2%}'}))




# --- Prediction ---
if submit:
    if land_size == 0 or building_size == 0:
        st.warning("Luas tanah dan bangunan tidak boleh nol!")
    else:
        # Persiapkan input untuk model
        input_dict = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'land_size_m2': land_size,
            'building_size_m2': building_size,
            'floors': floors,
            'building_age': building_age,
            'garages': garages,
            'property_type': encode_map['property_type'].get(property_type, 0),
            'furnishing': encode_map['furnishing'].get(furnishing, 1),
            'property_condition': encode_map['property_condition'].get(property_condition, 2)
        }

        data_input = pd.DataFrame([input_dict])[features]

        with st.spinner("🔄 Memproses prediksi..."):
            harga_prediksi = model.predict(data_input)[0]
            harga_rupiah = f"Rp {harga_prediksi:,.0f}".replace(",", ".")

        st.divider()
        st.subheader("💰 Estimasi Harga Rumah")
        
        # Box hasil prediksi
        st.markdown(
            f"""
            <div class="result-box">
                <h2>🏡 {harga_rupiah}</h2>
                <p>Perkiraan berdasarkan data properti yang Anda masukkan</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Cari Properti Serupa ---
        st.subheader("🔍 Properti Serupa")
        
        try:
            # Cari properti dengan kriteria yang mirip
            similar_properties = df[
                (df['bedrooms'].between(bedrooms-1, bedrooms+1)) &
                (df['bathrooms'].between(bathrooms-1, bathrooms+1)) &
                (df['land_size_m2'].between(land_size*0.8, land_size*1.2)) &
                (df['building_size_m2'].between(building_size*0.8, building_size*1.2)) &
                (df['floors'] == floors)
            ]
            
            if not similar_properties.empty:
                # Ambil 3 properti terdekat
                similar_properties['price_diff'] = abs(similar_properties['price_in_rp'] - harga_prediksi)
                similar_properties = similar_properties.sort_values('price_diff').head(3)
                
                for _, prop in similar_properties.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="similar-property">
                            <h4>{prop['title']}</h4>
                            <p>📍 {prop['address']}</p>
                            <p>🛏️ {int(prop['bedrooms'])} Kamar | 🚿 {int(prop['bathrooms'])} Kamar Mandi</p>
                            <p>📐 Luas Tanah: {prop['land_size_m2']} m² | Luas Bangunan: {prop['building_size_m2']} m²</p>
                            <p>💰 <strong>Rp {prop['price_in_rp']:,.0f}</strong></p>
                            <a href="{prop['url']}" target="_blank">🔗 Lihat Detail Properti</a>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Tidak ditemukan properti serupa dalam database kami.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mencari properti serupa: {str(e)}")

        # --- Evaluasi Model ---
        if metrics:
            st.subheader("📊 Evaluasi Model")
            st.success(f"**Akurasi Pada Score Prediksi (R² Score):** {metrics['R2'] * 100:.2f}%")

               # --- PDF Export ---
        st.subheader("📥 Unduh Hasil Prediksi")
        pdf_html = f"""
        <html>
        <head>
            <style>
                @page {{
                    margin: 0;
                }}
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    padding: 0;
                    margin: 0;
                    position: relative;
                }}
                .yellow-stripe {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 25px;
                    height: 100%;
                    background-color: #FFD700;
                    z-index: -1;
                }}
                .content {{
                    padding: 30px 40px 30px 60px;
                    position: relative;
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #4f8bf9;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                ul {{
                    margin-left: 20px;
                    padding-left: 0;
                }}
                li {{
                    margin-bottom: 8px;
                    position: relative;
                    padding-left: 15px;
                }}
                li:before {{
                    content: "•";
                    color: #FFD700;
                    font-size: 20px;
                    position: absolute;
                    left: 0;
                    top: -2px;
                }}
                .highlight {{
                    color: #4f8bf9;
                    font-weight: bold;
                    font-size: 1.1em;
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }}
                .logo {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .date {{
                    color: #666;
                    font-size: 14px;
                }}
                .price-box {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                    text-align: center;
                    border-left: 5px solid #FFD700;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 15px;
                    border-top: 1px solid #eee;
                    font-size: 12px;
                    color: #777;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="yellow-stripe"></div>
            <div class="content">
                <div class="header">
                    <div class="logo">🏠 Prediksi Rumah</div>
                    <div class="date">{datetime.now().strftime("%d %B %Y")}</div>
                </div>
                
                <h2>Estimasi Harga Properti</h2>
                
                <div class="price-box">
                    <h3 style="margin:0;color:#2c3e50;">Estimasi Harga</h3>
                    <p style="font-size:24px;margin:10px 0;color:#4f8bf9;"><strong>{harga_rupiah}</strong></p>
                </div>
                
                <h3>Detail Properti:</h3>
                <ul>
                    <li><b>Jumlah Kamar Tidur:</b> {bedrooms}</li>
                    <li><b>Jumlah Kamar Mandi:</b> {bathrooms}</li>
                    <li><b>Luas Tanah:</b> {land_size} m²</li>
                    <li><b>Luas Bangunan:</b> {building_size} m²</li>
                    <li><b>Jumlah Lantai:</b> {floors}</li>
                    <li><b>Usia Bangunan:</b> {building_age} tahun</li>
                    <li><b>Jumlah Garasi:</b> {garages}</li>
                    <li><b>Perabotan:</b> {furnishing.capitalize()}</li>
                    <li><b>Kondisi:</b> {property_condition.capitalize()}</li>
                </ul>
                
                <div class="footer">
                    <p>Dokumen ini dibuat secara otomatis oleh Aplikasi Prediksi Harga Rumah</p>
                    <p>© {datetime.now().year} Jasasaja Rumah 123 | All rights reserved</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        pdf_file = convert_html_to_pdf(pdf_html)
        if pdf_file:
            st.download_button(
                label="📄 Unduh Hasil Prediksi (PDF)",
                data=pdf_file.getvalue(),
                file_name="hasil_prediksi_harga_rumah.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Gagal membuat file PDF.")

# --- Footer ---
st.markdown("<div class='footer'>© 2023 Aplikasi Prediksi Harga Rumah | Jasasaja Rumah 123</div>", unsafe_allow_html=True)