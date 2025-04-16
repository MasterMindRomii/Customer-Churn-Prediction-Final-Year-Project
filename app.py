import streamlit as st
import pandas as pd
import numpy as np
import base64
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to set background image
def set_background(image_path):
    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image not found. Using default background.")

# Set page title
st.set_page_config(page_title="Imbalanced Data Handler", layout="wide")

# # Optional background
# bg_image_path = "D:/Romi - Personal Docs/MPR FINAL YEAR PROJECT/Customer-Churn-Prediction-Final-Year-Project/6155818.jpg"
# set_background(bg_image_path)

# st.title("Handle Imbalanced Data with SMOTEENN")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of the Dataset:", df.head())

    # Select target column
    target_column = st.selectbox("Select Target Column", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Check class distribution
        st.write("Class Distribution Before Resampling:")
        st.bar_chart(y.value_counts())
        st.write("Original dataset shape:", X.shape)

        # Drop non-numeric columns
        X = X.select_dtypes(include=['number'])

        if X.shape[1] == 0:
            st.error("No numeric features found. Please check your data.")
        else:
            try:
                # SMOTEENN with milder strategy
                smote_enn = SMOTEENN(sampling_strategy=0.5)
                X_resampled, y_resampled = smote_enn.fit_resample(X, y)

                if X_resampled.shape[0] == 0:
                    raise ValueError("SMOTEENN produced an empty dataset.")

                st.write("Resampled data shape:", X_resampled.shape)
                st.write("Class Distribution After SMOTEENN Resampling:")
                st.bar_chart(pd.Series(y_resampled).value_counts())

            except Exception as e:
                st.warning(f"SMOTEENN failed: {e}")
                st.info("Trying fallback method: SMOTE (only oversampling)...")

                try:
                    smote = SMOTE(sampling_strategy='auto')
                    X_resampled, y_resampled = smote.fit_resample(X, y)

                    st.write("Resampled data shape (SMOTE only):", X_resampled.shape)
                    st.write("Class Distribution After SMOTE Resampling:")
                    st.bar_chart(pd.Series(y_resampled).value_counts())

                except Exception as smote_error:
                    st.error(f"SMOTE also failed: {smote_error}")
                    st.stop()

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Show results
            st.write("Model Performance")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.success("Data Resampling & Model Training Completed!")
