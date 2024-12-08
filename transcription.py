import whisper
import numpy as np
import librosa
import io
import streamlit as st

class WhisperTranscriptionApp:
    def __init__(self):
        # Load the Whisper model
        self.model = whisper.load_model("base")
        
        # Initialize session state for transcription
        if 'transcription' not in st.session_state:
            st.session_state.transcription = ""

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data using Whisper model.
        """
        try:
            result = self.model.transcribe(audio_data)
            return result['text']
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return ""

    def _convert_audio_to_numpy(self, audio_data: bytes) -> np.ndarray:
        """
        Convert audio data to numpy array using librosa.
        """
        try:
            # Handle UploadedFile objects from Streamlit
            if hasattr(audio_data, "read"):  # Check if it's a file-like object
                audio_data = audio_data.read()  # Extract raw bytes
            
            # Create a temporary file to save the audio data
            with io.BytesIO(audio_data) as audio_file:
                # Use librosa to load the audio file
                audio_np, sample_rate = librosa.load(
                    audio_file, 
                    sr=16000,  # Resample to 16kHz (Whisper's preferred sample rate)
                    mono=True  # Convert to mono
                )
            
            return audio_np
        
        except Exception as e:
            st.error(f"Audio conversion error: {str(e)}")
            return None
