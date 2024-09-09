import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import io

def automated_eda(df):
    # Display the first few rows of the dataframe
    st.write("First few rows of the dataset:")
    st.write(df.head())
    
    # Display basic information about the dataframe
    st.write("Dataframe Info:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    # Display summary statistics
    st.write("Summary Statistics:")
    st.write(df.describe(include='all'))
    
    # Display correlation matrix
    st.write("Correlation Matrix:")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 0:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)
    else:
        st.write("Datatypes not suitable for this visualization")
    
    # Display pairplot horizontally if possible
    st.write("Pairplot:")
    if numeric_df.shape[1] > 0:
        pairplot_col1, pairplot_col2 = st.columns(2)
        with pairplot_col1:
            st.write("Numerical Columns Pairplot:")
        with pairplot_col2:
            fig = sns.pairplot(numeric_df)
            st.pyplot(fig)
    else:
        st.write("Datatypes not suitable for this visualization")
    
    # Display distribution of each column in two columns (horizontally)
    st.write("Distribution of each column:")
    if numeric_df.shape[1] > 0:
        cols = st.columns(2)  # Create two columns for the histograms
        for i, column in enumerate(numeric_df.columns):
            fig, ax = plt.subplots()
            sns.histplot(numeric_df[column], kde=True, ax=ax)
            
            # Alternate plotting between the two columns
            if i % 2 == 0:
                with cols[0]:
                    st.write(f"Distribution of {column}")
                    st.pyplot(fig)
            else:
                with cols[1]:
                    st.write(f"Distribution of {column}")
                    st.pyplot(fig)
    else:
        st.write("Datatypes not suitable for this visualization")
