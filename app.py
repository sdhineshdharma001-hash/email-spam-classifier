import streamlit as st
import pickle

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Page settings
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧"
)

st.title("📧 Email Spam Classifier")
st.write("Enter a message below to check whether it is Spam or Ham.")

# Message input
message = st.text_area(
    "Enter your email/message:",
    placeholder="Type your message here..."
)

# Prediction
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        message_vectorized = vectorizer.transform([message])
        prediction = model.predict(message_vectorized)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
            st.write("This message is likely to be spam.")
        else:
            st.success("✅ HAM / LEGITIMATE MESSAGE")
            st.write("This message appears to be legitimate.")