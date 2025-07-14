# 🎬 IMDb Sentiment Analyzer
### A web app that performs real-time sentiment analysis on movie reviews, deployed and accessible online.

This project demonstrates a full, end-to-end data science workflow. It begins with data sourcing and preparation, moves through model development and hyperparameter tuning, and concludes with the deployment of the final model as an interactive web application using Streamlit.

## 🚀 Live Application

### **[➡️ Launch the App: imdb-sentiment-prediction.streamlit.app](https://imdb-sentiment-prediction.streamlit.app)**

## 🌟 Key Features

-   **Real-Time Classification**: Instantly classify any movie review as **Positive** or **Negative**.
-   **Probabilistic Confidence**: Displays the model's confidence score for each prediction.
-   **Interactive & User-Friendly UI**: A clean and simple interface built with Streamlit for ease of use.
-   **Optimized Performance**: Powered by a Logistic Regression model that has been tuned for optimal performance using `GridSearchCV`.

## 🎯 Project Motivation & Business Value

In the entertainment industry, understanding audience reception is critical for success. Movie studios, streaming platforms, and marketing agencies invest heavily in gauging public opinion to inform marketing campaigns, predict box office success, and guide future content creation.

This project tackles that need by providing an automated tool to instantly analyze the sentiment of user-generated text, such as reviews from IMDb, Rotten Tomatoes, or social media. This project serves as a proof-of-concept for a valuable business intelligence tool.

## 🧠 The Data Science Workflow

The entire model development process is documented in the [**`model_training_and_evaluation.ipynb`**](model_training_and_evaluation.ipynb) notebook. The key stages are outlined below.

### 1. Data Sourcing and Preparation
-   **Dataset**: The model was trained on the well-known **IMDb Large Movie Review Dataset**, containing 50,000 labeled movie reviews.
-   **Preprocessing**: The data was loaded and structured into a Pandas DataFrame. The labels were mapped from `0/1` to human-readable "Negative"/"Positive" classes.
-   **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets to ensure a fair and unbiased evaluation of the final model.

### 2. Model Development and Tuning
-   **Model Choice**: A **Logistic Regression** classifier was selected for this task. It offers a strong balance of high performance, fast training times, and good interpretability, making it an excellent choice for a production-ready model.
-   **Feature Extraction**: Text reviews were converted into numerical features using a `TfidfVectorizer`, which reflects how important a word is to a review while down-weighting common words.
-   **Hyperparameter Tuning**: `GridSearchCV` was used to systematically search for the optimal settings for the entire pipeline. This included tuning `ngram_range` and `min_df` for the vectorizer, and the regularization parameter `C` for the Logistic Regression model to prevent overfitting and maximize performance.

### 3. Final Model Evaluation
The final, tuned model was evaluated on the held-out test set, achieving an **accuracy of ~90.8%**. This strong performance indicates its ability to generalize well to new, unseen movie reviews.

## 🛠️ Technical Stack

-   **Core Language**: Python 3.11
-   **Data Manipulation**: Pandas
-   **Machine Learning Pipeline**: Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `Pipeline`, `GridSearchCV`)
-   **Web Framework**: Streamlit
-   **Deployment**: Streamlit Community Cloud
-   **Development Environment**: Google Colab

## 🖥️ How to Run Locally

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/hynesc/IMDb-sentiment
    cd imdb-sentiment
    ```

2.  **Set Up a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the App**:
    ```bash
    streamlit run app.py
    ```
