import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import random

nltk.download('vader_lexicon')
api_google_key= ""
# Initialize the NLTK Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch YouTube comments
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor

# Optimized function for batch translation
def batch_translate_text(api_google_key, texts, target_language="en", batch_size=100):
    translate_service = build("translate", "v2", developerKey=api_google_key)
    translated_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = translate_service.translations().list(
            q=batch,
            target=target_language
        ).execute()
        translated_texts.extend(
            [item["translatedText"] for item in response.get("translations", [])]
        )
    return translated_texts

# Parallelized translation using multithreading
def parallel_translate(api_google_key, texts, target_language="en", max_workers=5):
    batch_size = 100
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                batch_translate_text,
                api_google_key,
                texts[i:i + batch_size],
                target_language
            )
            for i in range(0, len(texts), batch_size)
        ]
        results = [future.result() for future in futures]
    return [text for sublist in results for text in sublist]


import random

# Updated fetch_comments function
def fetch_comments(api_key, video_id, max_comments=500):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=min(100, max_comments - len(comments))
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments  # Return all fetched comments (up to max_comments)

def translate_text(api_google_key, text, target_language="en"):
    translate_service = build("translate", "v2", developerKey=api_google_key)
    response = translate_service.translations().list(
        q=text,
        target=target_language
    ).execute()
    return response["translations"][0]["translatedText"]


def fetch_trending_videos(api_key, region_code="US", category="Now", max_results=500):
    category_map = {"Now": None, "Music": "10", "Games": "20", "Movie": "30"}
    youtube = build("youtube", "v3", developerKey=api_key)
    
    # Set the category ID based on the selected category
    category_id = category_map.get(category)

    response = youtube.videos().list(
        part="snippet",
        chart="mostPopular",
        regionCode=region_code,
        videoCategoryId=category_id,  # Can be None or empty for "Now" category
        maxResults=max_results
    ).execute()

    videos = []
    for item in response.get("items", []):
        videos.append({
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"]
        })
    return videos




# Function to analyze comments for sentiment using NLTK
def analyze_comments(comments):
    data = []
    for comment in comments:
        sentiment_scores = sia.polarity_scores(comment)
        sentiment = "positive" if sentiment_scores['compound'] > 0 else "negative" if sentiment_scores['compound'] < 0 else "neutral"
        data.append({
            "comment": comment,
            "sentiment": sentiment,
            "sentiment_score": sentiment_scores['compound']
        })

    return pd.DataFrame(data)
def fetch_video_details(api_key, video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    if response.get("items"):
        video_data = response["items"][0]["snippet"]
        return {
            "title": video_data["title"],
            "thumbnail_url": video_data["thumbnails"]["high"]["url"]
        }
    return None

# Main Streamlit app
def main():
    RED_LOGO_URL = "pages/yt.png" 
    st.image(RED_LOGO_URL, width=100)
    st.title("YouTube Comment Sentiment Analysis App")

    api_key = ""

    video_source = st.radio("Select Source", ["Trending Videos", "Video URL"], horizontal=True)

    if video_source == "Trending Videos":
        # List of countries where YouTube is available
        youtube_countries = {
            "United States": "US", "India": "IN", "United Kingdom": "GB", 
            "Canada": "CA", "Australia": "AU", "Germany": "DE", "France": "FR", 
            "Japan": "JP", "South Korea": "KR", "Brazil": "BR", "Mexico": "MX",
            "Italy": "IT", "Russia": "RU", "Netherlands": "NL", "South Africa": "ZA",
            "Saudi Arabia": "SA", "Turkey": "TR", "Sweden": "SE", "Indonesia": "ID",
            "Argentina": "AR", "Spain": "ES", "Thailand": "TH", "Vietnam": "VN",
            "Philippines": "PH", "Malaysia": "MY", "Poland": "PL", "Singapore": "SG",
            "New Zealand": "NZ", "Ireland": "IE", "Switzerland": "CH", "Belgium": "BE"
        }

        # Dropdown for selecting a country
        selected_country = st.selectbox("Select Country", list(youtube_countries.keys()))
        region_code = youtube_countries[selected_country]

        category = st.selectbox("Select Category", ["Now", "Music", "Games", "Movie"])
        st.info("Fetching trending videos...")
        trending_videos = fetch_trending_videos(api_key, region_code, category)

        if trending_videos:
            st.subheader("Trending Videos")
            for video in trending_videos:
                st.image(video["thumbnail_url"], width=300)
                st.write(f"**{video['title']}**")
                if st.button(f"Analyze Comments for {video['title']}"):
                    video_id = video["video_id"]
                    analyze_video(api_key, video_id)
        else:
            st.warning("No trending videos found.")

    elif video_source == "Video URL":
        video_url = st.text_input("Enter YouTube Video URL")
    if st.button("Analyze"):
        try:
            video_id = video_url.split("v=")[-1].split("&")[0]
            video_details = fetch_video_details(api_key, video_id)

            if video_details:
                st.image(video_details["thumbnail_url"], width=500)
                st.subheader(video_details["title"])

                analyze_video(api_key, video_id)
            else:
                st.error("Failed to fetch video details. Please check the URL.")
        except IndexError:
            st.error("Invalid YouTube URL format.")


def analyze_video(api_key, video_id):
    st.info("Fetching comments...")
    comments = fetch_comments(api_key, video_id, max_comments=500)

    if not comments:
        st.warning("No comments found for this video.")
        return

    st.success(f"Fetched the comments for this video.")

    # Select a random subset of comments for analysis
    random_subset = random.sample(comments, min(len(comments), 200))  # Analyze a random 200 comments

    # Translate the random subset of comments to English
    st.info("Translating comments to English...")
    try:
        translated_comments = parallel_translate(api_google_key, random_subset, target_language="en")
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return

    # Analyze translated comments
    st.info("Analyzing sentiment of the translated comments...")
    analysis_df = analyze_comments(translated_comments)

    # Sentiment Distribution
    st.header("Sentiment Distribution")
    sentiment_counts = analysis_df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", color=["green", "red", "blue"], ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Number of Comments")
    ax.set_xticklabels(sentiment_counts.index, rotation=0)
    st.pyplot(fig)

    # Actionable Insights
    st.header("Actionable Insights")

    total_comments = len(analysis_df)
    positive_comments = len(analysis_df[analysis_df["sentiment"] == "positive"])
    negative_comments = len(analysis_df[analysis_df["sentiment"] == "negative"])
    neutral_comments = len(analysis_df[analysis_df["sentiment"] == "neutral"])

    if positive_comments > 0:
        st.success(f"🌟 Positive Insight: {positive_comments}/{total_comments} comments are positive. Keep up the good work! Users are engaging positively, indicating a strong connection to the content.")
    else:
        st.warning("⚠️ Positive Insight: No positive comments found. Consider creating more engaging or inspiring content.")

    if negative_comments > 0:
        st.error(f"🚨 Negative Insight: {negative_comments}/{total_comments} comments are negative. Address any concerns or criticism raised by viewers to improve content perception.")
    else:
        st.success("✅ Negative Insight: No negative comments detected. Maintain your current quality and tone to avoid negative feedback.")

    if neutral_comments > 0:
        st.info(f"📊 Neutral Insight: {neutral_comments}/{total_comments} comments are neutral. While engagement is moderate, consider ways to encourage stronger emotional responses to your content.")

    # Comments with Sentiment
    st.header("Comments with Sentiment")
    for _, row in analysis_df.iterrows():
        st.markdown(f"<p><b>Comment:</b> {row['comment']}</p>", unsafe_allow_html=True)
        st.write(f"**Sentiment:** {row['sentiment']} (Score: {row['sentiment_score']:.2f})")
        st.write("---")

    # Downloadable CSV
    st.download_button(
        label="Download Analysis as CSV",
        data=analysis_df.to_csv(index=False),
        file_name="analysis.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
