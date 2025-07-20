# IMDb Sentiment Analyzer

A web app that performs real-time sentiment analysis on movie reviews. This project demonstrates a full, end-to-end data science workflow, from data sourcing and preparation to model tuning and final deployment as an interactive application.

---
## Live Application

You can view the deployed interactive application here:
**[https://imdb-sentiment-prediction.streamlit.app](https://imdb-sentiment-prediction.streamlit.app)**

---
## Key Features

-   **Real-Time Classification**: Instantly classify any movie review as **Positive** or **Negative**.
-   **Probabilistic Confidence**: Displays the model's confidence score for each prediction.
-   **Interactive & User-Friendly UI**: A clean and simple interface built with Streamlit for ease of use.
-   **Optimized Performance**: Powered by a Logistic Regression model tuned for optimal performance.

---
## 1. Problem Statement

In the entertainment industry, understanding audience reception is critical for success. Movie studios, streaming platforms, and marketing agencies need to efficiently gauge public opinion to inform marketing campaigns, predict box office success, and guide content strategy.

This project tackles that need by providing an automated tool to instantly analyze the sentiment of user-generated text. It serves as a proof-of-concept for a valuable business intelligence tool that can derive insights from reviews on IMDb, Rotten Tomatoes, or social media.

---
## 2. Dataset

The model was trained on the well-known **"Large Movie Review Dataset"** from IMDb. It contains 50,000 highly polar movie reviews that have been pre-labeled for sentiment (Positive/Negative). This balanced dataset is ideal for training a binary classification model.

---
## 3. Methodology

The project follows a standard data science workflow, documented in the accompanying Jupyter Notebook.

### a. Data Preprocessing
-   The dataset was loaded and structured into a Pandas DataFrame.
-   The sentiment labels were mapped from a binary format (`0/1`) to human-readable classes ("Negative"/"Positive").
-   The data was split into training (80%) and testing (20%) sets to ensure an unbiased evaluation of the final model.

### b. Feature Engineering and Model Training
-   **Feature Extraction**: Text reviews were converted into a numerical format using a `TfidfVectorizer`. This technique represents text based on word frequency while down-weighting common words that appear across all reviews (e.g., "the", "a", "is"), making it effective for identifying sentiment-bearing words.
-   **Model Selection**: A **Logistic Regression** classifier was chosen. It provides a strong balance of high performance and high interpretability, making it an excellent baseline and a suitable choice for production.

### c. Hyperparameter Tuning and Evaluation
-   A `GridSearchCV` pipeline was implemented to systematically find the optimal hyperparameters for both the vectorizer and the model. This process tuned:
    -   `ngram_range` and `min_df` for the `TfidfVectorizer`.
    -   The regularization parameter `C` for the `LogisticRegression` model to prevent overfitting.
-   The final, tuned model was evaluated on the held-out test set, achieving an **accuracy of ~90.8%**, indicating it generalizes well to new, unseen movie reviews.

---
## 4. File Structure
-   `run.py`: The main Streamlit application script that loads the trained model and serves the interactive UI.
-   `model_training_and_evaluation.ipynb`: A Jupyter Notebook containing all steps for data analysis, feature engineering, model training, and evaluation.
-   `imdb_model.joblib`: The pre-trained and tuned model.
-   `requirements.txt`: A list of all necessary Python libraries to ensure a reproducible environment.
-   `README.md`: This file.

---
## 5. How to Run Locally

1.  **Clone the Repository**:
    ```bash
    git clone github.com/hynesc/imdb-sentiment
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

---
## 6. Deployment

This application is deployed on Streamlit Community Cloud. The `requirements.txt` file is crucial for a successful deployment, as it instructs the cloud service which libraries and versions to install to perfectly match the development environment.

---
## 7. Tools and Libraries

-   **Data Science**: Pandas, Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `Pipeline`, `GridSearchCV`)
-   **Web Application & Deployment**: Streamlit, Streamlit Community Cloud
-   **Core Language**: Python 3.11
-   **Development Environment**: Google Colab / Jupyter Notebook
