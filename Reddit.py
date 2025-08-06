# Streamlit Application for Subreddit Sentiment Analysis

import streamlit as st
import praw
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download("vader_lexicon")
nltk.download("punkt")

# Matplotlib and Seaborn settings
sns.set(style="darkgrid", context="talk", palette="Dark2")

# Reddit API credentials
REDDIT_CLIENT_ID = ""
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT = ""

# Function to fetch subreddit posts
def fetch_subreddit_posts(subreddit_name, limit=100):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    headlines = []
    for submission in reddit.subreddit(subreddit_name).new(limit=limit):
        headlines.append(submission.title)
    return headlines

# Function to perform sentiment analysis
def analyze_sentiment(headlines):
    sia = SIA()
    results = []
    for line in headlines:
        pol_score = sia.polarity_scores(line)
        pol_score["headline"] = line
        results.append(pol_score)
    df = pd.DataFrame.from_records(results)
    df["label"] = 0
    df.loc[df["compound"] > 0.2, "label"] = 1
    df.loc[df["compound"] < -0.2, "label"] = -1
    return df

# Streamlit app starts here
# Add Reddit logo
RED_LOGO_URL = "pages/redditlogo.jpeg"  # Use a valid Reddit logo URL
st.image(RED_LOGO_URL, width=100)
st.title("Subreddit Sentiment Analysis")
st.markdown("Analyze the sentiment of posts from your favorite subreddit using NLTK's VADER Sentiment Analyzer.")

# Input subreddit details
st.header("Subreddit Configuration")
subreddit_name = st.text_input("Enter Subreddit Name", "politics")
post_limit = st.slider("Number of Posts to Analyze", 10, 500, 100)

# Analyze button
if st.button("Analyze"):
    try:
        with st.spinner(f"Fetching posts from r/{subreddit_name}..."):
            headlines = fetch_subreddit_posts(subreddit_name, limit=post_limit)

        st.success(f"Fetched {len(headlines)} posts from r/{subreddit_name}.")

        with st.spinner("Analyzing sentiment..."):
            df = analyze_sentiment(headlines)

        st.success("Sentiment analysis completed.")

        # Display results
        st.header("Sentiment Distribution")
        counts = df["label"].value_counts(normalize=True) * 100
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts, ax=ax)
        ax.set_xticklabels(["Negative", "Neutral", "Positive"])
        ax.set_ylabel("Percentage")
        st.pyplot(fig)

        st.header("Sample Headlines")
        st.subheader("Positive Headlines")
        st.write(df[df["label"] == 1].headline.head(5).to_list())

        st.subheader("Negative Headlines")
        st.write(df[df["label"] == -1].headline.head(5).to_list())

        # Save results to a CSV file
        st.download_button(
            label="Download Sentiment Analysis CSV",
            data=df.to_csv(index=False),
            file_name=f"{subreddit_name}_sentiment_analysis.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("Developed by Watt Warriors ⚡")
