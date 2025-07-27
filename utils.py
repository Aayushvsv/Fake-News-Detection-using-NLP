# backend/utils.py

import re
from typing import List, Optional
import logging

def clean_text(text: str) -> str:
    """
    Perform basic cleaning of input text: lowercase, remove non-alphanumerics, etc.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces/newlines with single space
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation/specials
    return text.strip()

def is_valid_input(text: Optional[str]) -> bool:
    """
    Validate that input text is a non-empty string.
    """
    return isinstance(text, str) and bool(text.strip())

def tokenize(text: str) -> List[str]:
    """
    Splits text into words. Extend for more complex tokenization as needed.
    """
    # (For real-world use, see NLTK or spaCy)
    return text.split()

def log_exception(msg: str, ex: Exception):
    """
    Log formatted exception with message.
    """
    logging.error(f"{msg}: {ex}", exc_info=True)

# Example extension: function to compute prediction confidence
def get_top_confidence(prob_array):
    """
    Returns the maximum probability from the model's predict_proba output.
    """
    if not hasattr(prob_array, "max"):  # check for numpy/arraylike
        raise ValueError("Input must be an array-like with .max()")
    return float(prob_array.max())
