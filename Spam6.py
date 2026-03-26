import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_data():
    data_path = 'spam.tsv'
    if not os.path.exists(data_path):
        st.error(f"{data_path} not found. Please ensure the file is in the current directory.")
        return None
    df = pd.read_csv(data_path, sep="\t")
    hamDf = df[df['label'] == "ham"]
    spamDf = df[df['label'] == "spam"]
    hamDf = hamDf.sample(spamDf.shape[0])  # Balancing the dataset
    return pd.concat([hamDf, spamDf], ignore_index=True)

# Train model function
@st.cache_resource
def train_model(finalDf):
    X_train, X_test, Y_train, Y_test = train_test_split(
        finalDf['message'], finalDf['label'], test_size=0.2, random_state=0, stratify=finalDf['label']
    )
    
    # Initialize the model pipeline
    model = Pipeline([('tfidf', TfidfVectorizer()), ('model', RandomForestClassifier(n_estimators=100, n_jobs=-1))])
    model.fit(X_train, Y_train)
    
    # Make predictions and evaluate accuracy
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Save model
    joblib.dump(model, "spam_classifier_model.pkl")
    
    return model, accuracy

# Load model function
def load_model():
    model_path = "spam_classifier_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found. Please train the model first.")
        return None

# Main function for Streamlit app
def main():
    st.title("Email Spam Detector")
    st.write("Classify email messages as spam or not spam using a machine learning model.")

    # Load data
    data = load_data()
    if data is None:
        return  # Exit if data is not loaded

    # Train or load model
    if st.button("Train Model"):
        model, accuracy = train_model(data)
        st.success("Model trained successfully!")
        st.write("**Accuracy:**", accuracy)
    else:
        model = load_model()

    # Predict on new message
    st.write("### Test the Model with Your Own Message")
    user_input = st.text_area("Enter an email message:")
    if st.button("Classify"):
        if model is not None and user_input:
            prediction = model.predict([user_input])
            st.write("**Prediction:**", "Spam" if prediction[0] == "spam" else "Not Spam")
        else:
            st.warning("Please train the model or enter a message to classify.")

if __name__ == "__main__":
    main()
