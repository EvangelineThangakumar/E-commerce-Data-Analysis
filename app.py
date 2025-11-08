#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pickle

# Load pickled files
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("segment_labels.pkl", "rb") as f:
    segment_labels = pickle.load(f)
with open("item_similarity.pkl", "rb") as f:
    item_similarity_df = pickle.load(f)
with open("products.pkl", "rb") as f:
    product_list = pickle.load(f)

st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.sidebar.title("HOME")
page = st.sidebar.radio("Choose a Module:", ["Customer Segmentation", "Product Recommendation"])

# Customer Segmentation

if page == "Customer Segmentation":
    st.header("Customer Segmentation (RFM Prediction)")
    recency = st.number_input("Recency (days since last purchase):", min_value=1)
    frequency = st.number_input("Frequency (number of purchases):", min_value=1)
    monetary = st.number_input("Monetary (total spend):", min_value=100.0)

    if st.button("Predict Cluster"):
        try:
            input_data = np.array([[recency, frequency, monetary]])
            log_data = np.log1p(input_data)
            scaled_data = scaler.transform(log_data)
            cluster = kmeans.predict(scaled_data)[0]
            segment = segment_labels.get(cluster, "Unknown")

            st.success(f"Predicted Segment: **{segment}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Product Recommendation Page

elif page == "Product Recommendation":
    st.header("Product Recommendation")
    product_name = st.text_input("Enter a product name:")

    if st.button("Get Recommendations"):
        if product_name not in item_similarity_df.columns:
            st.warning(f"Product '{product_name}' not found in database.")
        else:
            similar_scores = item_similarity_df[product_name].sort_values(ascending=False)
            top_products = similar_scores.iloc[1:6].index.tolist()
            st.subheader("Recommended Products:")
            for i, prod in enumerate(top_products, 1):
                st.write(f"{i}. {prod}")

