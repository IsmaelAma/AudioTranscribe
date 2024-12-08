import streamlit as st
import torch
from file_transcription import file_transcription
from microphone_transcription import microphone_transcription

# Set page config as the first Streamlit command
st.set_page_config(page_title="Audio Transcription and Summarization App", page_icon="🎙️")
st.title("Audio Transcription and Summarization")
def main():
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 File Upload", "🎤 Microphone Input"])

    if torch.cuda.is_available():
        # st.sidebar.success("CUDA is available. GPU acceleration enabled!")
        print("CUDA is available. GPU acceleration enabled!")
    else:
        # st.sidebar.warning("CUDA not available. Running on CPU.")
        print("CUDA not available. Running on CPU.")

    with tab1:
        file_transcription()  # Handle file upload transcription
    
    with tab2:
        microphone_transcription()  # Handle microphone input transcription

if __name__ == "__main__":
    main()
