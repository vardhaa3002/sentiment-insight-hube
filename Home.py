import streamlit as st

# Sentimental Insights Hub 🌟
st.title("Sentimental Insights Hub 🌟")

# Slide 2: Problem Statement 🧠
st.header("Problem Statement 🧠")
st.subheader("Sentiment Analysis Track")
st.markdown("""
**Problem Statement:**  
Build an app that analyzes posts and comments to gauge community sentiment. The solution should:  
- Perform **real-time sentiment analysis**.  
- Identify **trending topics**.  
- Generate **sentiment reports**.  
""")

# Slide 3: Proposed Solution 💡
st.header("Proposed Solution 💡")
st.subheader("Proposed Solution: Sentimental Insights Hub")
st.markdown("""
- **Unified Platform:** Combines sentiment analysis capabilities across multiple data sources (web pages, Reddit posts, and YouTube comments).  
- **Real-Time Analysis:** Provides instant sentiment insights by leveraging APIs and machine learning models.  
- **Sentiment Reports:** Generates detailed visualizations and downloadable reports.  
""")

# Slide 4: Process Flow 🔄
st.header("Process Flow 🔄")
st.subheader("Process Flow Diagram")
st.markdown("""
### Input Data Source:  
- 🌐 Web pages  
- 📄 Reddit posts  
- 🎥 YouTube comments  

### Data Extraction:  
- 🛠️ Scraping content using APIs and BeautifulSoup.  

### Language Detection & Translation:  
- 🌍 Google Translate API to detect the language and translate to English.  

### Sentiment Analysis:  
- 🔍 Hugging Face Transformers and NLTK for sentiment classification (Positive, Negative, Neutral).  

### Trending Topics Identification:  
- 📈 Analyze word frequency and context to detect popular topics.  

### Visualization & Reporting:  
- 📊 Use Matplotlib to create sentiment distribution charts.  
- 📥 Generate downloadable sentiment analysis reports.  
""")

# Footer
st.markdown("---")
st.markdown("Developed by Watt Warriors")
