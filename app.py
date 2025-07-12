#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and data
kmeans = joblib.load('rfm_kmeans.pkl')
scaler = joblib.load('rfm_scaler.pkl')
product_similarity = joblib.load('product_similarity.pkl')
product_lookup = joblib.load('product_lookup.pkl')  # DataFrame with index=StockCode, column=Description

# 1. Product Recommendation Module
st.title("E-Commerce Intelligence Dashboard")

st.header("Product Recommendation")

product_input = st.text_input("Enter Product Code (e.g., 84029E)")

if st.button("Get Recommendations"):
    if product_input in product_similarity.columns:
        similar = product_similarity[product_input].sort_values(ascending=False)[1:6]
        st.subheader("Top 5 Similar Products:")
        for i, code in enumerate(similar.index):
            name = product_lookup.loc[code, 'Description']
            st.markdown(f"**{i+1}. {name}**  \n(Product Code: `{code}`)", unsafe_allow_html=True)
    else:
        st.error("Product not found. Please check the product code.")

# 2️ Customer Segmentation Module
st.header("Customer Segmentation")

r = st.number_input("Recency (in days)", min_value=0, step=1)
f = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
m = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0)

if st.button("Predict Cluster"):
    input_scaled = scaler.transform([[r, f, m]])
    cluster = kmeans.predict(input_scaled)[0]

    # Segment labeling based on interpretation
    def map_cluster_to_segment(c):
        mapping = {
            0: 'High-Value',
            1: 'Regular',
            2: 'Occasional',
            3: 'At-Risk'
        }
        return mapping.get(c, 'Unknown')

    segment = map_cluster_to_segment(cluster)
    st.success(f"Predicted Segment: **{segment}** (Cluster {cluster})")


# In[ ]:




