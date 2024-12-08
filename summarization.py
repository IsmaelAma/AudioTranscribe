from transformers import BartForConditionalGeneration, BartTokenizer
import streamlit as st

class SummarizationApp:
    def __init__(self):
        # Load the BART model for summarization
        self.bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # Initialize session state for summary
        if 'summary' not in st.session_state:
            st.session_state.summary = ""

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
