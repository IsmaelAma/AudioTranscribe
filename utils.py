import streamlit as st
import numpy as np

def handle_file_upload():
    """
    Handles file upload and returns the audio data as numpy array.
    """
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'm4a'])

    if uploaded_file is not None:
        return uploaded_file.read()
    return None

def handle_local_file():
    """
    Handles the local file loading for testing purposes.
    """
    try:
        local_file_path = "assets/test_audio.mp3"  # Update this path as needed
        with open(local_file_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Test file not found at: {local_file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading test file: {str(e)}")
        return None
