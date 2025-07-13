import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer # Needed for the pipeline
from sklearn.linear_model import LogisticRegression # Needed for the pipeline

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# --- MODEL LOADING ---
# We use a function with a cache decorator to load the model only once.
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained model from a .joblib file."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct directory.")
        return None

# IMPORTANT: Make sure the path matches the name of your .joblib file
MODEL_FILE = 'best_model.joblib'
model = load_model(MODEL_FILE)


# --- APP INTERFACE ---
st.title("🎬 IMDb Sentiment Analyzer")
st.markdown("""
Welcome! This app uses a **Logistic Regression** model to predict whether a movie review is **Positive** or **Negative**.
The model was trained on 50,000 reviews from the IMDb dataset.

**Enter a review below and see the magic happen!**
""")

# User input form
with st.form(key='review_form'):
    user_input = st.text_area("Enter your movie review here:", "This movie was absolutely fantastic! The acting, the plot, everything was perfect.", height=150)
    submit_button = st.form_submit_button(label='Analyze Sentiment ✨')


# --- PREDICTION LOGIC ---
if submit_button and model is not None:
    if user_input.strip():
        # The model's pipeline expects a list of documents
        prediction = model.predict([user_input])
        probability = model.predict_proba([user_input])

        # Display the result with a nice card layout
        st.subheader("Analysis Result")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == "Positive":
                st.success("🎉 Positive Review")
                # Get the probability of the 'Positive' class
                confidence = probability[0][1]
            else:
                st.error("😞 Negative Review")
                # Get the probability of the 'Negative' class
                confidence = probability[0][0]
        
        with col2:
            st.metric(label="Confidence", value=f"{confidence:.0%}")

        st.info("The 'Confidence' score represents the model's predicted probability for the detected sentiment.")
    else:
        st.warning("Please enter a review to analyze.")

# Add a footer
st.markdown("---")
st.markdown("Built using scikit-learn and Streamlit.")
