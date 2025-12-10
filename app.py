import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

# ----------------------------
# 1. Load data & train model
# ----------------------------

@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])
    y = pd.Series(cancer["target"], name="target")
    df = pd.concat([X, y], axis=1)
    return cancer, df, X, y

@st.cache_resource
def train_model(X, y):
    # Same split & model as your notebook: SVC without scaling/tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    model = SVC()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True)

    return model, X_train, X_test, y_train, y_test, y_test_pred, acc, cm, report


# ----------------------------
# 2. Streamlit UI
# ----------------------------

def main():
    st.set_page_config(
        page_title="Breast Cancer Classification",
        layout="wide",
    )

    st.title("Cancer Classification")
    st.write(
        """
        This app uses the **Breast Cancer Wisconsin (Diagnostic)** dataset  
        and an **SVM classifier (SVC)** without tuning to predict whether a tumor
        is **Malignant (0)** or **Benign (1)**.
        """
    )

    # Load data & model
    cancer, df, X, y = load_data()
    model, X_train, X_test, y_train, y_test, y_test_pred, acc, cm, report = train_model(
        X, y
    )

    tab1, tab2, tab3 = st.tabs(["📊 Dataset & EDA", "📈 Model Performance", "🔮 Make Predictions"])

    # ----------------------------
    # Tab 1: Dataset & EDA
    # ----------------------------
    with tab1:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", df.shape[0])
        col2.metric("Features", df.shape[1] - 1)
        col3.metric("Target Classes", 2)

        st.write("**Target classes:** 0 = Malignant, 1 = Benign")

        st.markdown("### Sample Data")
        st.dataframe(df.head())

        st.markdown("### Target Distribution")
        target_counts = df["target"].value_counts().rename({0: "Malignant", 1: "Benign"})
        st.bar_chart(target_counts)

        st.markdown("### Correlation (First 10 Features + Target)")
        corr_cols = list(df.columns[:10]) + ["target"]
        corr_matrix = df[corr_cols].corr()
        st.write("Correlation Matrix (No Matplotlib)")
        st.dataframe(corr_matrix)


    # ----------------------------
    # Tab 2: Model Performance
    # ----------------------------
    with tab2:
        st.subheader("Model: Support Vector Classifier (SVC) – Default Parameters")
        st.write("No scaling, no hyperparameter tuning – same as your base model.")

        st.metric("Accuracy on Test Set", f"{acc*100:.2f}%")

        st.markdown("### Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["True Malignant (0)", "True Benign (1)"],
            columns=["Pred Malignant (0)", "Pred Benign (1)"],
        )
        st.table(cm_df)

        st.markdown("### Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))

    # ----------------------------
    # Tab 3: Make Predictions
    # ----------------------------
    with tab3:
        st.subheader("Predict Tumor Type")

        mode = st.radio(
            "Choose input method:",
            ["Use a sample from test set", "Enter feature values manually"],
        )

        if mode == "Use a sample from test set":
            st.write(
                "Select a sample index from the test set to see the model prediction."
            )
            idx = st.slider(
                "Test sample index",
                min_value=0,
                max_value=len(X_test) - 1,
                value=0,
            )

            sample_X = X_test.iloc[idx]
            true_y = y_test.iloc[idx]
            pred_y = model.predict(sample_X.values.reshape(1, -1))[0]

            st.markdown("#### Feature Values (Test Sample)")
            st.dataframe(sample_X.to_frame(name="Value"))

            st.markdown("#### Result")
            st.write(f"**True Label:** {'Malignant (0)' if true_y == 0 else 'Benign (1)'}")
            st.write(f"**Predicted Label:** {'Malignant (0)' if pred_y == 0 else 'Benign (1)'}")

        else:
            st.write(
                "Enter feature values below. Defaults are set to the dataset means."
            )

            input_values = []
            with st.form("manual_input_form"):
                for feature in X.columns:
                    col_min = float(X[feature].min())
                    col_max = float(X[feature].max())
                    col_mean = float(X[feature].mean())

                    val = st.slider(
                        feature,
                        min_value=col_min,
                        max_value=col_max,
                        value=col_mean,
                        step=(col_max - col_min) / 100.0,
                    )
                    input_values.append(val)

                submitted = st.form_submit_button("Predict")

            if submitted:
                sample = np.array(input_values).reshape(1, -1)
                pred_y = model.predict(sample)[0]

                st.markdown("#### Prediction")
                if pred_y == 0:
                    st.success("Model prediction: **Malignant (0)**")
                else:
                    st.success("Model prediction: **Benign (1)**")


if __name__ == "__main__":
    main()
