import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os
from transformers import pipeline
from google.cloud import speech_v1p1beta1 as speech



# Load environment variables
load_dotenv()

# Configure GenAI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Streamlit UI
st.title("YouTube Transcript Api")

# Function to extract transcript from YouTube video
def extract_transcript_details(youtube_video_url,chunk_size=2000):
    try:
       video_id = youtube_video_url.split("=")[1]
       transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
       transcript = ""
       for i in transcript_text:
        transcript += " " + i["text"]
        # Divide transcript into chunks
        transcript_chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
       return transcript_chunks
       
    except Exception as e:
       st.error(f"Error occurred: {str(e)}")
    return None

# Function to extract transcript from YouTube video using ASR
def extract_transcript_with_asr(video_url):
    try:
        client = speech.SpeechClient()
        audio_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            language_code="en-US",
        )
        audio = speech.RecognitionAudio(url=video_url)
        response = client.recognize(config=audio_config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcript
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None


def generate_summary(text):
    try:
       summarizer = pipeline("summarization")
       summary = summarizer(text, max_length=250, min_length=30, do_sample=False)
       return summary[0]['summary_text']
    except Exception as e:
      st.error(f"Error occurred while generating summary: {str(e)}")
      return None

# Get user input for YouTube video URL
youtube_video_url = st.text_input("Enter YouTube Video URL", "")

# Process user input and display transcript
if st.button("Extract Transcript"):
 if youtube_video_url:
   st.info("Fetching transcript... Please wait.")
   transcript = extract_transcript_details(youtube_video_url)
   if transcript is None:  # If captions are not available, use ASR
            st.info("Transcript not found. Using Automatic Speech Recognition (ASR)...")
            transcript = extract_transcript_with_asr(youtube_video_url)
   if transcript:
    st.subheader("Transcript:")
    st.write(transcript)
    st.info("Generating summary... Please wait.")
    summary = generate_summary(transcript)
    if summary:
     st.subheader("Summary:")
     st.write(summary)
   else:
     st.warning("Please enter a valid YouTube video URL.")



