import streamlit as st
import pandas as pd 
import joblib as jb


model = jb.load("model.jb")
columns = jb.load("columns.jb")
df = pd.read_csv("laptops_simple.csv")

st.title("💻 Laptop Price Predictor")
st.write("Enter your laptop specifications to predict its estimated price.")

st.subheader("Laptop Specifications")

brand = st.selectbox("Brand", sorted(df["brand"].unique()))
category = st.selectbox("Category", sorted(df["category"].unique()))
processor = st.selectbox("Processor ", sorted(df["processor"].unique()))
graphics = st.selectbox("Graphics ", sorted(df["graphics"].unique()))
ram = st.selectbox("Ram (Gb)", sorted(df["ram"].unique()))
storage = st.selectbox("Storage (Gb)", sorted(df["storage"].unique()))
screen_size = st.selectbox("Screen size (inch)", sorted(df["screen_size"].unique()))

if st.button("Predict Laptop Price"):

    input_data = pd.DataFrame([{
        "brand": brand,
        "category": category,
        "processor": processor,
        "graphics": graphics,
        "ram": ram,
        "storage": storage,
        "screen_size": screen_size
    }])

    input_data = pd.get_dummies(
        input_data,
        columns = ["brand", "category", "processor", "graphics"])

    input_data = input_data.reindex(
        columns=columns, 
        fill_value=0)

    price = model.predict(input_data)[0]

    st.success(f"Estimated price: ₹ {price:,.2f}")




































































































































































 