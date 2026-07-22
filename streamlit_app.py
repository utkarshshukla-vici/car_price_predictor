import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st


def main():
    html_temp = """<h1 style='color: #2e6c80;'>Car Price Predictor</h1>"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("This app will predict the price of your car based on its features!")
    st.markdown("---")

    # Load Model
    model = xgb.XGBRegressor()
    model.load_model("xgb_model.json")

    # Input Fields (Exact features in exact order)
    st.subheader("Car Details")
    present_price = st.number_input("Present_Price", min_value=0.0, value=5.59, step=0.1)
    kms_driven = st.number_input("Kms_Driven", min_value=0, value=27000, step=100)
    fuel_type = st.selectbox("Fuel_Type", options=['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox("Seller_Type", options=['Dealer', 'Individual'])
    transmission = st.selectbox("Transmission", options=['Manual', 'Automatic'])
    owner = st.number_input("Owner", min_value=0, max_value=10, value=0, step=1)
    age = st.number_input("Age", min_value=0, max_value=50, value=10, step=1)

    if st.button("Predict"):
        # Preprocessing mappings exactly as in the notebook
        fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
        seller_mapping = {'Dealer': 0, 'Individual': 1}
        transmission_mapping = {'Manual': 0, 'Automatic': 1}

        # Create input dataframe in the exact order expected by the model
        data_new = pd.DataFrame({
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_mapping[fuel_type],
            'Seller_Type': seller_mapping[seller_type],
            'Transmission': transmission_mapping[transmission],
            'Owner': owner,
            'Age': age
        }, index=[0])

        # Predict
        prediction = model.predict(data_new)
        st.success(f"Estimated Car Selling Price: {prediction[0]:.2f}")


if __name__ == "__main__":
    main()