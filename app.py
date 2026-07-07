import streamlit as st
import numpy as np
import joblib

model=joblib.load("rf_model.pkl")

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💻 Laptop Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 6px solid #4CAF50;'>
    <h4>📊 Estimate your laptop price!</h4>
    <p>Enter values for <b>processor speed</b>, <b>RAM size</b>, and <b>storage capacity</b> to predict the approximate price of a laptop using a trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px dashed #aaa;'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    processor_speed = st.number_input("🔧 Processor Speed (GHz)", value=2.50, step=0.50, min_value=0.5)
    ram_size = st.number_input("🧠 RAM Size (GB)", value=16, step=8, min_value=4)

with col2:
    storage_capacity = st.number_input("💾 Storage Capacity (GB)", value=512, step=256, min_value=128)

X=[processor_speed,ram_size,storage_capacity]
st.markdown("<hr style='border: 1px dashed #aaa;'>", unsafe_allow_html=True)

prediction = st.button("🎯 Click for Price Estimate",use_container_width=True)

if prediction:
    st.balloons()
    x1=np.array(X)
    prediction=model.predict([x1])[0]
    st.markdown("<h3 style='color:#4CAF50;'>✅ Estimated Price</h3>", unsafe_allow_html=True)
    
    st.metric(label="💰 Laptop Price (in ₹)", value=f"₹{prediction:,.2f}")
    
    st.success("Prediction successful! Adjust inputs to try again.")
else:
    st.write("Please use the button for getting a prediction") 

st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>© 2026 Laptop Price Predictor by Daksh Jain</p>", unsafe_allow_html=True)