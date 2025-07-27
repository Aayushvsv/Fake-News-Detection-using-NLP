#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
python -m nltk.downloader wordnet

# Download spaCy model
python -m spacy download en_core_web_sm