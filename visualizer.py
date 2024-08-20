# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import numpy as np
# import io

# def automated_eda(df):
#     # Display the first few rows of the dataframe
#     st.write("First few rows of the dataset:")
#     st.write(df.head())
    
#     # Display basic information about the dataframe
#     st.write("Dataframe Info:")
#     buffer = io.StringIO()
#     df.info(buf=buffer)
#     s = buffer.getvalue()
#     st.text(s)
    
#     # Display summary statistics
#     st.write("Summary Statistics:")
#     st.write(df.describe(include='all'))
    
#     # Display correlation matrix
#     st.write("Correlation Matrix:")
#     numeric_df = df.select_dtypes(include=[np.number])
#     if numeric_df.shape[1] > 0:
#         corr = numeric_df.corr()
#         fig, ax = plt.subplots()
#         sns.heatmap(corr, annot=True, ax=ax)
#         st.pyplot(fig)
#     else:
#         st.write("Datatypes not suitable for this visualization")
    
#     # Display pairplot
#     st.write("Pairplot:")
#     if numeric_df.shape[1] > 0:
#         fig = sns.pairplot(numeric_df)
#         st.pyplot(fig)
#     else:
#         st.write("Datatypes not suitable for this visualization")
    
#     # Display distribution of each column
#     st.write("Distribution of each column:")
#     if numeric_df.shape[1] > 0:
#         for column in numeric_df.columns:
#             fig, ax = plt.subplots()
#             sns.histplot(numeric_df[column], kde=True, ax=ax)
#             st.pyplot(fig)
#     else:
#         st.write("Datatypes not suitable for this visualization")

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
    
    # Create a horizontal scrolling container for visualizations
    st.write("Visualizations (scroll horizontally to view all):")
    visualization_container = st.beta_container()
    
    with visualization_container:
        # Use columns to create a horizontal layout
        cols = st.columns(4)  # Adjust the number of columns as needed
        
        # Display correlation matrix
        with cols[0]:
            st.write("Correlation Matrix:")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 0:
                corr = numeric_df.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
                st.pyplot(fig)
            else:
                st.write("Datatypes not suitable for this visualization")
        
        # Display distribution of each column
        if numeric_df.shape[1] > 0:
            for i, column in enumerate(numeric_df.columns, start=1):
                with cols[i % 4]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(numeric_df[column], kde=True, ax=ax)
                    plt.title(f'Distribution of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Count')
                    st.pyplot(fig)
        else:
            st.write("Datatypes not suitable for this visualization")
        
        # Display pairplot
        with cols[-1]:
            st.write("Pairplot:")
            if numeric_df.shape[1] > 0:
                fig = sns.pairplot(numeric_df)
                st.pyplot(fig)
            else:
                st.write("Datatypes not suitable for this visualization")

    # Add custom CSS for horizontal scrolling
    st.markdown("""
    <style>
        .stHorizontalBlock {
            overflow-x: auto;
            white-space: nowrap;
        }
        .stHorizontalBlock > div {
            display: inline-block;
            vertical-align: top;
            margin-right: 20px;
        }
    </style>
    """, unsafe_allow_html=True)