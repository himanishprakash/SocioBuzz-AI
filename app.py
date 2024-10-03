import os
import streamlit as st
from apify_client import ApifyClient
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from textblob import TextBlob
from openai import OpenAI
import base64
from io import BytesIO

# Initialize the OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
apify_api_key = os.getenv('apify_api_key')
client = OpenAI(api_key=openai_api_key)

# Set up YouTube API client
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=youtube_api_key)

# Function to convert Matplotlib plots to base64 images
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

# Function to fetch video links based on a keyword
def fetch_videos_by_keyword(keyword, max_results):
    search_response = youtube.search().list(
        q=keyword,
        type='video',
        part='id,snippet',
        maxResults=max_results
    ).execute()

    video_links = [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in search_response['items']]
    return video_links

# Function to fetch video links for a channel ID
def fetch_videos_by_channel_id(channel_id, max_results):
    search_response = youtube.search().list(
        channelId=channel_id,
        type='video',
        part='id,snippet',
        maxResults=max_results,
        order='date'
    ).execute()

    video_links = [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in search_response['items']]
    return video_links

def get_rss_feed(youtube_links):
    all_docs = []
    for link in youtube_links:
        loader = YoutubeLoader.from_youtube_url(link)
        docs = loader.load()
        all_docs.extend(docs)
    
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)
    split_docs = text_splitter.split_documents(all_docs)
    
    return split_docs

def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def summarize_text_with_gpt4(text_chunks):
    summaries = []
    for chunk in text_chunks:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You have been assigned to summarize the following text and do sentiment analysis."},
                {"role": "user", "content": f"Summarize the following text:\n\n{chunk}"}
            ]
        )
        summaries.append(response.choices[0].message.content.strip())
    
    combined_summary = " ".join(summaries)
    return combined_summary




# Function to get sentiment using OpenAI API for Instagram
def get_sentiment(comment):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to analyze sentiment."},
            {"role": "user", "content": f"Analyze the sentiment of this comment: '{comment}' and respond with Positive, Negative, or Neutral."}
        ]
    )
    sentiment = response.choices[0].message.content.strip()
    if sentiment not in ["Positive", "Negative", "Neutral"]:
        sentiment = "Neutral"
    return sentiment

# Function to summarize the analysis using OpenAI API for Instagram
def summarize_analysis(data_summary, table_data):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to summarize analyses."},
            {"role": "user", "content": f"Provide a summary of the Instagram posts analysis based on the sentiment scores and number of likes over time. The data summary is: {data_summary}. Here is the table of post data:\n{table_data}."},
            {"role": "user", "content": "Additionally, provide a brief summary of what people are talking about negatively and identify the main topics of concern."},
            {"role": "user", "content": "Can you also suggest strategies to improve the sentiment scores of future posts based on this analysis?"}
        ]
    )
    summary = response.choices[0].message.content.strip()
    return summary

def get_summary(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a smart reviewer designed to summarize text."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
            {"role": "user", "content": "Provide a brief summary of the text."},
            {"role": "user", "content": "What are the key points in the text? highlight the main ideas."}
        ]
    )
    summary = response.choices[0].message.content.strip()
    return summary

def main():
    st.title("SocioBuzz: Social Media Analysis Tool")

    app_mode = st.selectbox("Choose the app", ["Instagram", "YouTube"])

    if app_mode == "Instagram":
        st.subheader("Instagram Posts Analysis")

        profile_url = st.text_input("Enter the Instagram profile URL:")
        num_posts = st.number_input("Enter the number of posts to analyze:", min_value=1, max_value=50, value=10)

        if st.button("Analyze"):
            if profile_url and num_posts:
                apify_client = ApifyClient(apify_api_key)

                run_input = {
                    "directUrls": [profile_url],
                    "resultsType": "posts",
                    "resultsLimit": num_posts,
                    "searchType": "hashtag",
                    "searchLimit": 1,
                    "addParentData": False,
                }

                run = apify_client.actor("shu8hvrXbJbY3Eb9W").call(run_input=run_input)

                results = []
                for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
                    results.append(item)

                with open('instagram_posts.json', 'w') as json_file:
                    json.dump(results, json_file, indent=4)

                with open('instagram_posts.json', 'r') as file:
                    data = json.load(file)

                total_comments_count = 0
                total_likes_count = 0
                all_comments_texts = []
                post_dates = []
                post_data = []
                sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

                for post in data:
                    total_comments_count += post["commentsCount"]
                    total_likes_count += post["likesCount"]
                    post_comments = [comment["text"] for comment in post.get("latestComments", [])]
                    all_comments_texts.extend(post_comments)
                    post_dates.append(post["timestamp"])

                    sentiment_scores = []
                    for comment in post_comments:
                        sentiment = get_sentiment(comment)
                        sentiment_scores.append(sentiment)
                        if sentiment in sentiment_counts:
                            sentiment_counts[sentiment] += 1
                        else:
                            sentiment_counts["Neutral"] += 1

                    post_data.append({
                        "Post Date": post["timestamp"],
                        "Comments": post["commentsCount"],
                        "Likes": post["likesCount"],
                        "Sentiment Scores": sentiment_scores
                    })

                avg_comments_per_post = total_comments_count / num_posts
                avg_likes_per_post = total_likes_count / num_posts

                df = pd.DataFrame(post_data)
                df["Post Date"] = pd.to_datetime(df["Post Date"])
                df = df.sort_values("Post Date")

                all_comments = " ".join(all_comments_texts)
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                img1_b64 = fig_to_base64(fig1)

                data_summary = f"Comments: {total_comments_count}, Likes: {total_likes_count}, Avg comments/post: {avg_comments_per_post:.2f}, Avg likes/post: {avg_likes_per_post:.2f}"
                table_data = df.to_string(index=False)

                final_summary = summarize_analysis(data_summary, table_data)

                st.subheader("Sentiment Analysis and Summary")
                if "Positive" in final_summary:
                    st.markdown(f"<span style='color: green;'>{final_summary}</span>", unsafe_allow_html=True)
                else:
                    st.write(final_summary)

                st.subheader("Visualizations")
                st.image(f"data:image/png;base64,{img1_b64}", caption="Sentiment Analysis of Instagram Comments")

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.imshow(wordcloud, interpolation='bilinear')
                ax2.axis('off')
                st.image(f"data:image/png;base64,{fig_to_base64(fig2)}", caption="Word Cloud of Instagram Comments")

                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.plot(df["Post Date"], df["Likes"], marker='o', linestyle='-')
                ax3.set_xlabel("Date")
                ax3.set_ylabel("Number of Likes")
                ax3.set_title("Number of Likes Over Time")
                st.image(f"data:image/png;base64,{fig_to_base64(fig3)}", caption="Number of Likes Over Time")

                st.subheader("Average Comments and Likes per Post")
                st.write(f"Average Comments per Post: {avg_comments_per_post:.2f}")
                st.write(f"Average Likes per Post: {avg_likes_per_post:.2f}")

                st.subheader("Detailed Data for Each Post")
                st.dataframe(df)

                st.download_button(label='Download JSON data', data=json.dumps(results, indent=4), file_name='instagram_posts.json')

                st.subheader("Key Data Points for Social Influencers")
                st.write("- **Total Comments and Likes:** Understand the overall engagement.")
                st.write("- **Average Comments and Likes per Post:** Helps in assessing the performance of individual posts.")
                st.write("- **Sentiment Analysis and Summary:** Provides a brief summary of the sentiment of comments, including top positive and negative comments.")
                st.write("- **Word Cloud of Comments:** Provides insights into the common themes and feedback from followers.")
                st.write("- **Number of Likes Over Time:** Shows the trend of engagement over time.")

            else:
                st.write("Please enter a profile URL and the number of posts to start the analysis.")

    elif app_mode == "YouTube":
        st.subheader("YouTube Video Fetcher")

        option = st.selectbox('Search by:', ('Keyword', 'Channel ID'))

        video_links = []
        if option == 'Keyword':
            keyword = st.text_input('Enter a keyword:')
            max_results = st.number_input('Enter the number of videos to fetch:', min_value=1, max_value=50, value=20)
            if st.button('Fetch Videos'):
                if keyword:
                    video_links = fetch_videos_by_keyword(keyword, max_results)
                    st.write(f'Top {max_results} Videos:')
                    for link in video_links:
                        st.write(link)

        elif option == 'Channel ID':
            channel_id = st.text_input('Enter a channel ID:')
            max_results = st.number_input('Enter the number of videos to fetch:', min_value=1, max_value=50, value=20)
            if st.button('Fetch Videos'):
                if channel_id:
                    video_links = fetch_videos_by_channel_id(channel_id, max_results)
                    st.write(f'Top {max_results} Videos:')
                    for link in video_links:
                        st.write(link)

        if len(video_links) > 0:
            rss_feed = get_rss_feed(video_links)
            combined_text = " ".join([doc.page_content for doc in rss_feed])

            text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=0)
            text_chunks = text_splitter.split_text(combined_text)

            st.write("RSS Feed Summary:")
            summary = summarize_text_with_gpt4(text_chunks)
            summarizer = get_summary(summary)
            
            if "Positive" in summarizer:
                st.markdown(f"<span style='color: green;'>{summarizer}</span>", unsafe_allow_html=True)
            else:
                st.write(summarizer)

            st.write("Sentiment Analysis:")
            polarity, subjectivity = perform_sentiment_analysis(summarizer)
            st.write(f"Polarity: {polarity}, Subjectivity: {subjectivity}")

            sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

            if polarity > 0:
                sentiments['Positive'] += 1
            elif polarity == 0:
                sentiments['Neutral'] += 1
            else:
                sentiments['Negative'] += 1

            labels = sentiments.keys()
            sizes = sentiments.values()
            colors = ['green', 'blue', 'red']

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

            st.pyplot(fig)

if __name__ == "__main__":
    main()
