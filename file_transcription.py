import streamlit as st
from transcription import WhisperTranscriptionApp
from utils import handle_file_upload, handle_local_file
from summarization import SummarizationApp
import numpy as np

def file_transcription():
    """
    Handle file upload transcription with a 'Use Local File' button for testing.
    """
    st.subheader("Upload Audio File")
    
    # File uploader
    audio_data = handle_file_upload()
    
    # Use Local File button
    use_local_file = st.button("Use Local File", key="use_local_file_button")
    
    if use_local_file:
        audio_data = handle_local_file()
    
    if audio_data is not None:
        # Convert audio to numpy array
        transcription_app = WhisperTranscriptionApp()
        audio_np = transcription_app._convert_audio_to_numpy(audio_data)
        
        # Transcribe the audio
        transcription_file = transcription_app.transcribe_audio(audio_np)
        st.session_state.transcription = transcription_file
        
        # Display transcription
        st.write("### Transcription:")
        st.text_area("Transcribed Text", transcription_file, height=200)

        # Summarization button
        if st.button("Summarize", key="summarize_button_file"):
            summarization_app = SummarizationApp()
            summary = summarization_app.summarize_text(transcription_file)
            st.session_state.summary = summary

            # Display the summary
            st.write("### Summary:")
            st.text_area("Summarized Text", summary, height=200)
