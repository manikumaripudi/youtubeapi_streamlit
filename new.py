import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import google.generativeai as gen_ai
import os
import hashlib
import uuid

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)
gemini_model = gen_ai.GenerativeModel("gemini-pro")
# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-t5-base')

# Streamlit UI
st.title("YouTube Transcript Similarity Search using Gemini-Pro")

# Function to extract transcript from YouTube video
def extract_transcript_details(youtube_video_url, chunk_size=2000):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        # Divide transcript into chunks
        transcript_chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
        return transcript_chunks
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None

# Function to perform similarity search
def similar_search(query_data):
    try:
        # Encode the query chunk
        query_embedding = model.encode([query_data])[0]

        # Perform similarity search with distance metric
        search_results = client.search(
            collection_name="genai-docs",
            query_vector=query_embedding.tolist(), # Convert to list
            limit=2, # Retrieve top 2 similar chunks
        )

        return search_results
    except Exception as e:
        st.error(f"Error occurred during similarity search: {str(e)}")
        return None
#Entering the input as url
youtube_video_url = st.text_input("Enter YouTube Video URL", "")
if youtube_video_url:
    #creation of embeddings and insertion of transcription chuncks into Qdrant db.
    try:
        # Get transcript of the YouTube video
        video_id = youtube_video_url.split("=")[-1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([chunk['text'] for chunk in transcript_text])

        # Divide transcript into chunks
        chunks = [transcript[i:i+2000] for i in range(0, len(transcript), 2000)]

        # Initialize lists for storing document IDs and hashed chunks
        doc_ids = []
        hashed_chunks = set()

        # Iterate over chunks
        for chunk in chunks:
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            if chunk_hash not in hashed_chunks:
                hashed_chunks.add(chunk_hash)

                # Encode chunk using SentenceTransformer
                embedding = model.encode([chunk])
                embeds = embedding[0]

                # Generate a random ID
                doc_id = str(uuid.uuid4())

                # Upsert the document into the collection
                client.upsert(
                    collection_name="genai-docs",
                    points=[
                        models.PointStruct(
                            id=doc_id,
                            vector=embeds,
                            payload={
                                "URL": youtube_video_url,
                                'data': chunk,
                            },
                        ),
                    ],
                )

                # Append the document ID to the list
                doc_ids.append(doc_id)
       
    
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
else:
    st.warning("Please enter a YouTube video URL.")

 #Allow the user to ask questions based on the provided URL


question = st.chat_input("Ask a question based on the provide URL")

if  question:
    # Extract transcript chunks from the YouTube video URL
    transcript_chunks = extract_transcript_details(youtube_video_url)

    # Perform similarity search based on the user's question
    search_results = similar_search(question)
    total_chunks = ""

    if search_results:
        for result in search_results[:2]:
            total_chunks = total_chunks + "\n" + "\n" + result.payload["data"]

        # Generate a prompt for the conversational AI
        prompt = f"Question: {question} \n\n Data: {total_chunks} \n\n Instruction: Based on the above data, if the question has an answer in the data, then provide me the answer. Otherwise, give me any answer related to the question."

        # Initialize or retrieve the chat session
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = gemini_model.start_chat(history=[])

        gemini_response = st.session_state.chat_session.send_message(prompt)

        # Display Gemini's response
        st.markdown(gemini_response.text)
