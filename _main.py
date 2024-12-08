import streamlit as st
import whisper
import torch
import io
import numpy as np
import librosa
from typing import Optional, Tuple
from transformers import BartForConditionalGeneration, BartTokenizer

# Set page config as the first Streamlit command
st.set_page_config(page_title="Audio Transcription and Summarization App", page_icon="🎙️")

class WhisperTranscriptionApp:
    def __init__(self):
        # Load the Whisper model
        self.model = whisper.load_model("base")
        
        # Load the BART model for summarization
        self.bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # Initialize the app title
        st.title("🎙️ Audio Transcription and Summarization")
        
        # Initialize session state for transcription and summary
        if 'transcription' not in st.session_state:
            st.session_state.transcription = ""
        if 'summary' not in st.session_state:
            st.session_state.summary = ""

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

    def summarize_text(self, text: str) -> str:
        """
        Summarize text using BART model.
        """
        try:
            inputs = self.bart_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=100000, truncation=True)
            summary_ids = self.bart_model.generate(inputs, max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            return self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"Summarization error: {e}")
            return ""

    def file_transcription(self):
        """
        Handle file upload transcription with a 'Use Local File' button for testing.
        """
        st.subheader("Upload Audio File")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'm4a'])

        # Use Local File button
        use_local_file = st.button("Use Local File", key="use_local_file_button")
        
        if uploaded_file is not None:
            # Read the uploaded file
            audio_bytes = uploaded_file.read()
            audio_np = self._convert_audio_to_numpy(audio_bytes)
            
        elif use_local_file:
            # Load the local test file
            try:
                local_file_path = "assets/test_audio.mp3"  # Update this path as needed
                with open(local_file_path, "rb") as f:
                    audio_bytes = f.read()
                audio_np = self._convert_audio_to_numpy(audio_bytes)
                st.success(f"Using local test file: {local_file_path}")
            except FileNotFoundError:
                st.error(f"Test file not found at: {local_file_path}")
                return
            except Exception as e:
                st.error(f"Error loading test file: {str(e)}")
                return
        else:
            st.info("Upload a file or click 'Use Local File' to proceed.")
            return

        # If audio data was successfully loaded
        if audio_np is not None:
            # Transcribe the audio
            transcription = self.transcribe_audio(audio_np)
            st.session_state.transcription = transcription
            
            # Display transcription
            st.write("### Transcription:")
            st.text_area("Transcribed Text", transcription, height=200)

            # Summarization button
            if st.button("Summarize", key="summarize_button_file"):
                summary = self.summarize_text(transcription)
                st.session_state.summary = summary
                
                # Display the summary
                st.write("### Summary:")
                st.text_area("Summarized Text", summary, height=200)
    
    def microphone_transcription(self):
        """
        Handle microphone input transcription.
        """
        st.subheader("Live Microphone Transcription")
        
        # Microphone input
        st.write("Record audio to transcribe:")
        recorded_audio = st.audio_input("Record Audio")
        
        if recorded_audio is not None:
            # Convert audio to numpy array
            audio_np = self._convert_audio_to_numpy(recorded_audio)
            
            if audio_np is not None:
                # Transcribe the audio
                transcription = self.transcribe_audio(audio_np)
                st.session_state.transcription = transcription
                
                # Display transcription
                st.write("### Transcription:")
                st.text_area("Transcribed Text", transcription, height=200)

                # Summarization button
                if st.button("Summarize", key="summarize_button_recorded"):
                    summary = self.summarize_text(transcription)
                    st.session_state.summary = summary
                    
                    # Display the summary
                    st.write("### Summary:")
                    st.text_area("Summarized Text", summary, height=200)

    def _convert_audio_to_numpy(self, audio_data):
        """
        Convert audio data to numpy array using librosa.
        """
        try:
            # Handle UploadedFile objects from Streamlit
            if hasattr(audio_data, "read"):  # Check if it's a file-like object
                audio_data = audio_data.read()  # Extract raw bytes
            
            # If audio_data is already a numpy array, return it
            if isinstance(audio_data, np.ndarray):
                return audio_data
            
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

    def run(self):
        """
        Run the Streamlit app.
        """
        # Check for CUDA availability
        if torch.cuda.is_available():
            # st.sidebar.success("CUDA is available. GPU acceleration enabled!")
            print("CUDA is available. GPU acceleration enabled!")
        else:
            # st.sidebar.warning("CUDA not available. Running on CPU.")
            print("CUDA not available. Running on CPU.")
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📁 File Upload", "🎤 Microphone Input"])
        
        with tab1:
            self.file_transcription()
        
        with tab2:
            self.microphone_transcription()

# Main app execution
def main():
    # Initialize and run the app
    app = WhisperTranscriptionApp()
    app.run()

if __name__ == "__main__":
    main()
