import streamlit as st
import pandas as pd
from data_processor import preprocess_data
from visualizer import automated_eda

def main():
    st.title("Project Insight")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the file
        df = pd.read_csv(uploaded_file)
        st.write("Data loaded successfully!")
        st.write("Raw Data:")
        st.write(df.head())

        try:
            # Preprocess data
            df_processed = preprocess_data(df)
            st.write("Processed Data:")
            st.write(df_processed.head())

            # Visualize data
            automated_eda(df_processed)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
