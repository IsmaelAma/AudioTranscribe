import streamlit as st
from transcription import WhisperTranscriptionApp
from summarization import SummarizationApp
import numpy as np

def microphone_transcription():
    """
    Handle microphone input transcription.
    """
    st.subheader("Live Microphone Transcription")
    
    # Microphone input
    st.write("Record audio to transcribe:")
    recorded_audio = st.audio_input("Record Audio")
    
    if recorded_audio is not None:
        # Convert audio to numpy array
        transcription_app = WhisperTranscriptionApp()
        audio_np = transcription_app._convert_audio_to_numpy(recorded_audio)
        
        if audio_np is not None:
            # Transcribe the audio
            transcription_recorded = transcription_app.transcribe_audio(audio_np)
            st.session_state.transcription = transcription_recorded
            
            # Display transcription
            st.write("### Transcription:")
            st.text_area("Transcribed Text", transcription_recorded, height=200)

            # Summarization button
            if st.button("Summarize", key="summarize_button_recorded"):
                summarization_app = SummarizationApp()
                summary = summarization_app.summarize_text(transcription_recorded)
                st.session_state.summary = summary
                
                # Display the summary
                st.write("### Summary:")
                st.text_area("Summarized Text", summary, height=200)
