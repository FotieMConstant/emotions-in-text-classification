# sentiment/ml_utils.py

from transformers import TFDistilBertModel, DistilBertTokenizer

# Load model and tokenizer
dbert_tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased')
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


def load_model():
    print(dbert_model.summary())  # Move the print statement here
    return dbert_model


def load_tokenizer():
    return dbert_tokenizer
