from pykospacing import Spacing
#from hanspell import spell_checker

from konlpy.tag import Mecab

import torch
import torch.nn as nn

from model.kobert import KoBERTforSequenceClassification
from kobert_transformers import get_tokenizer

import joblib


class model_loader():
    def __init__(self):
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(ctx)

        self.spacing = Spacing()
        
        # Mecab
        self.mecab = Mecab()

        # koBERT
        self.labelencoder = joblib.load('saves/labelencoder.pkl')
        self.kobert_tokenizer = get_tokenizer()

        self.checkpoint = torch.load('saves/kobert-emotion-text-classification.pth', map_location=device)
        self.classifier = KoBERTforSequenceClassification()
        self.classifier.load_state_dict(self.checkpoint['model_state_dict'])
        self.classifier.eval()

        # koGPT2


    def split_msg(msg):
        return self.spacing(msg)

    def tokenize_msg(msg):
        return self.mecab.morphs(msg)
        pass

    def classify_msg(msg):
        data = kobert_input(self.kobert_tokenizer, msg, device, 512)
        output = model(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit,dim=-1)
        softmax_logit = softmax_logit.squeeze()

        max_index = torch.argmax(softmax_logit).item()
        # max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

        category = labelencoder.inverse_transform([max_index])
        return category

    def reply_to_msg(msg):
        pass

    def kobert_input(tokenizer, str, device = None, max_seq_len = 512):
        index_of_words = tokenizer.encode(str)
        token_type_ids = [0] * len(index_of_words)
        attention_mask = [1] * len(index_of_words)

        # Padding Length
        padding_length = max_seq_len - len(index_of_words)

        # Zero Padding
        index_of_words += [0] * padding_length
        token_type_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        data = {
            'input_ids': torch.tensor([index_of_words]).to(device),
            'token_type_ids': torch.tensor([token_type_ids]).to(device),
            'attention_mask': torch.tensor([attention_mask]).to(device),
        }
        
        return data

    