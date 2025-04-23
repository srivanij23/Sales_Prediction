import streamlit as st
import pandas as pd
from src.modeling import train_models
from src.utils import save_predictions
from sklearn.model_selection import train_test_split

# Title
st.title("🚗 Car Purchase Amount Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview:", df.head())

    # Simple train-test split
    if st.button("Train Model and Predict"):
        X = df[['age', 'annual Salary']]  # adjust columns based on your dataset
        y = df['car purchase amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = train_model(X_train, y_train)

        predictions = model.predict(X_test)

        output_df = X_test.copy()
        output_df['car purchase amount'] = y_test
        save_predictions(output_df, predictions)

        st.success("✅ Model trained and predictions saved to 'outputs/predictions.csv'")
        st.write(output_df.head())
