import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # Needed for the pipeline
from sklearn.linear_model import LogisticRegression # Needed for the pipeline

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained model from a .joblib file."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct directory.")
        return None

# --- FEATURE IMPORTANCE FUNCTION ---
@st.cache_data
def get_feature_importance(_model):
    """Extracts and formats feature importances from the model pipeline."""
    # Extract the vectorizer and classifier from the pipeline
    vectorizer = _model.named_steps['tfidf']
    classifier = _model.named_steps['logreg']

    # Get feature names and their corresponding coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    # Create a DataFrame
    importance_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})

    # Get the top 15 positive and negative words
    top_positive = importance_df.sort_values(by='coefficient', ascending=False).head(15)
    top_negative = importance_df.sort_values(by='coefficient', ascending=True).head(15)

    return top_positive, top_negative


# Load the model
MODEL_FILE = 'best_logistic_regression_model.joblib'
model = load_model(MODEL_FILE)


# --- APP INTERFACE ---
st.title("🎬 IMDb Sentiment Analyzer")
st.markdown("""
Welcome! This app uses a **Logistic Regression** model to predict whether a movie review is **Positive** or **Negative**.
I trained this model on 50,000 reviews from the IMDb dataset.

Enter a review below and see the magic happen!
""")

# User input form
with st.form(key='review_form'):
    user_input = st.text_area("Enter your movie review here:", "This movie was fantastic! The acting, the plot, everything was perfect.", height=150)
    submit_button = st.form_submit_button(label='Analyze Sentiment ✨')


# --- PREDICTION LOGIC ---
if submit_button and model is not None:
    if user_input.strip():
        prediction = model.predict([user_input])
        probability = model.predict_proba([user_input])

        st.subheader("Analysis Result")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == "Positive":
                st.success("🎉 Positive Review")
                confidence = probability[0][1]
            else:
                st.error("😞 Negative Review")
                confidence = probability[0][0]
        
        with col2:
            st.metric(label="Confidence", value=f"{confidence:.0%}")

        st.info("The 'Confidence' score represents the model's predicted probability for the detected sentiment.")
    else:
        st.warning("Please enter a review to analyze.")

# --- MODEL INTERPRETABILITY SECTION ---
if model is not None:
    with st.expander("🤔 See how the model thinks"):
        st.markdown("""
        This chart shows the words that have the most influence on the model's predictions.
        The coefficients are the weights the model assigns to each word. A high positive value means the word is a strong indicator of a *Positive* review, while a high negative value indicates a *Negative* review.
        """)
        
        # Get and display feature importances
        top_pos, top_neg = get_feature_importance(model)

        # Plotting
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Positive Words")
            st.bar_chart(top_pos.set_index('feature')['coefficient'])
        
        with col2:
            st.subheader("Top Negative Words")
            # Invert the negative coefficients to make the bar chart positive
            st.bar_chart(top_neg.set_index('feature')['coefficient'].abs())

# Add a footer
st.markdown("---")
st.markdown("Built using scikit-learn and Streamlit.")