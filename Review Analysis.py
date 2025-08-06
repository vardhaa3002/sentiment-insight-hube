import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("Sentiment Analysis using Multiple Classifiers")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataset
    st.write("### Dataset Preview:")
    st.dataframe(data.head())

    # Preprocessing
    st.write("### Preprocessing:")
    data = data.dropna(subset=['reviewText', 'overall'])  # Drop rows with missing values in 'reviewText' or 'overall'

    # Convert 'overall' ratings to sentiment labels
    def label_sentiment(rating):
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'

    data['sentiment'] = data['overall'].apply(label_sentiment)

    # Display sentiment distribution
    st.write("#### Sentiment Distribution:")
    sentiment_counts = data['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # Feature Extraction (Bag of Words)
    vectorizer = CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(data['reviewText']).toarray()
    y = data['sentiment']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Support Vector Classifier': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': BernoulliNB(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    # Train and evaluate each classifier
    results = {}
    st.write("### Classifier Performance:")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        st.write(f"{name} Accuracy: {accuracy:.2f}")

    # Rank classifiers by accuracy
    ranked_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    st.write("### Ranked Classifiers:")
    for rank, (name, accuracy) in enumerate(ranked_results, 1):
        st.write(f"{rank}. {name} - Accuracy: {accuracy:.2f}")

    # Hyperparameter tuning
    st.write("### Hyperparameter Tuning:")
    best_params = {}
    tuned_results = {}
    for name, clf in classifiers.items():
        if name == 'Logistic Regression':
            param_grid = {'C': [0.1, 1, 10]}
        elif name == 'Decision Tree':
            param_grid = {'max_depth': [5, 10, 20]}
        elif name == 'Support Vector Classifier':
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        elif name == 'Random Forest':
            param_grid = {'n_estimators': [50, 100, 200]}
        elif name == 'Naive Bayes':
            param_grid = {'alpha': [0.1, 0.5, 1]}
        elif name == 'K-Nearest Neighbors':
            param_grid = {'n_neighbors': [3, 5, 10]}

        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_params[name] = grid_search.best_params_
        tuned_clf = grid_search.best_estimator_
        y_pred_tuned = tuned_clf.predict(X_test)
        tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
        tuned_results[name] = tuned_accuracy
        st.write(f"{name} Best Parameters: {grid_search.best_params_}, Tuned Accuracy: {tuned_accuracy:.2f}")

    # Compare Tuned Accuracies
    st.write("### Tuned Classifier Performance:")
    tuned_ranked_results = sorted(tuned_results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, accuracy) in enumerate(tuned_ranked_results, 1):
        st.write(f"{rank}. {name} - Tuned Accuracy: {accuracy:.2f}")

    # Word Frequency Visualization by Sentiment
    st.write("### Word Frequency by Sentiment (Unigrams, Bigrams, Trigrams):")
    sentiments = ['Positive', 'Neutral', 'Negative']

    for sentiment in sentiments:
        st.write(f"#### {sentiment} Sentiment:")
        sentiment_data = data[data['sentiment'] == sentiment]

        # Vectorize the review texts for the current sentiment
        sentiment_vectorizer = CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 3))
        sentiment_X = sentiment_vectorizer.fit_transform(sentiment_data['reviewText']).toarray()
        sentiment_word_freq = pd.DataFrame({
            'word': sentiment_vectorizer.get_feature_names_out(),
            'count': np.ravel(sentiment_X.sum(axis=0))
        }).sort_values(by='count', ascending=False).head(30)

        # Plot the frequencies for the sentiment
        fig, ax = plt.subplots()
        sns.barplot(x='count', y='word', data=sentiment_word_freq, ax=ax, palette='viridis')
        plt.title(f"Top 30 Words for {sentiment} Sentiment")
        st.pyplot(fig)


    st.write("Upload another file to analyze!")
