import streamlit as st
import requests
import matplotlib.pyplot as plt
from transformers import pipeline
from bs4 import BeautifulSoup

# Define API Keys (Replace with your actual keys)
GOOGLE_API_KEY = ""

# Load Hugging Face sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

def fetch_post_title(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for errors
        soup = BeautifulSoup(response.text, "html.parser")  
        
        # Assuming the post title is within <title> tag
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text().strip()
        else:
            return "Title not found"
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the post title: {e}")
        return None

# Function to detect and translate language using Google Translate API
def translate_with_google(text):
    try:
        url = f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_API_KEY}"
        data = {
            "q": text,
            "target": "en",
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        translated_text = result["data"]["translations"][0]["translatedText"]
        detected_language = result["data"]["translations"][0]["detectedSourceLanguage"]
        return translated_text, detected_language
    except requests.exceptions.RequestException as e:
        st.error(f"Error with Google Translate API: {e}")
        return text, "unknown"

# Function to perform sentiment analysis using Hugging Face
def analyze_sentiment_huggingface(text):
    try:
        sentiment = sentiment_analyzer(text)[0]
        return sentiment['label']
    except Exception as e:
        st.error(f"Error analyzing sentiment with Hugging Face: {e}")
        return "Unknown"

# Main Streamlit app
def main():
    st.set_page_config(page_title="Sentiment Analysis with Hugging Face", layout="wide")
    RED_LOGO_URL = "pages/kynlogo.jpeg" 
    st.image(RED_LOGO_URL, width=100)
    st.title("Sentiment Analysis on Kynhood")
    st.markdown("Analyze post titles fetched from URLs with language detection, translation, and sentiment analysis using Hugging Face and Google Translate.")

    # Input URL
    url = st.text_input("Enter the URL of the post:")

    if st.button("Analyze"):
        with st.spinner("Fetching title..."):
            title = fetch_post_title(url)
        
        if title:
            st.subheader("Post Title")
            st.write(f"**Extracted Title:** {title}")

            with st.spinner("Translating and detecting language..."):
                translated_title, detected_language = translate_with_google(title)
            
            st.write(f"**Detected Language:** {detected_language}")
            st.write(f"**Translated Title:** {translated_title}")

            with st.spinner("Analyzing sentiment..."):
                sentiment = analyze_sentiment_huggingface(translated_title)
            
            st.subheader("Sentiment Analysis")
            st.metric("Sentiment", sentiment)

            # Visualize sentiment (dummy scores for now)
            st.subheader("Sentiment Distribution")
            scores = {
                "POSITIVE": 0.5 if sentiment == "POSITIVE" else 0.2,
                "NEUTRAL": 0.5 if sentiment == "NEUTRAL" else 0.3,
                "NEGATIVE": 0.5 if sentiment == "NEGATIVE" else 0.1,
            }

            with st.container():
                # Custom HTML to adjust the size of the container
                st.markdown(
                    """
                    <style>
                    .custom-container {
                        max-width: 400px;
                        margin: 0 auto;
                        padding: 15px;
                    }
                    </style>
                    <div class="custom-container">
                    """,
                    unsafe_allow_html=True,
                )

                # Plot inside the styled container
                fig, ax = plt.subplots(figsize=(4.10, 3.10))  # Adjust the size as needed
                ax.bar(scores.keys(), scores.values(), color=["green", "gray", "red"])
                ax.set_title("Sentiment Distribution", fontsize=8, fontweight="bold")
                ax.set_ylabel("Score", fontsize=12)

                # Rotate x-axis labels to prevent overlap
                ax.set_xticks(range(len(scores.keys())))
                ax.set_xticklabels(scores.keys(), rotation=45, ha='right', fontsize=8)

                # Tight layout to prevent overlap
                plt.tight_layout()

                # Display the plot inside the container
                st.pyplot(fig, use_container_width=False)

                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <small>Powered by Watt_Warriors</small>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
