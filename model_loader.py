#from pykospacing import Spacing
#from hanspell import spell_checker

from konlpy.tag import Mecab

import torch
import torch.nn as nn

from model.kobert import KoBERTforSequenceClassification, kobert_input
from kobert_transformers import get_tokenizer

import joblib


class model_loader():
    def __init__(self):
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        #self.spacing = Spacing()
        
        # Mecab
        self.mecab = Mecab()

        # koBERT
        #self.classifier = joblib.load('saves/kobert_model.pkl')
        self.labelencoder = joblib.load('saves/labelencoder.pkl')
        self.kobert_tokenizer = get_tokenizer()

        self.checkpoint = torch.load('saves/kobert_model.pth', map_location=self.device)
        self.classifier = KoBERTforSequenceClassification()
        self.classifier.load_state_dict(self.checkpoint['model_state_dict'], strict=False)
        
        self.classifier.to(self.device)
        self.classifier.eval()

        # koGPT2


    def split_msg(self, msg):
        #return self.spacing(msg)
        pass

    def tokenize_msg(self, msg):
        return self.mecab.morphs(msg)

    def classify_msg(self, msg):
        data = kobert_input(self.kobert_tokenizer, msg, self.device, 512)
        output = self.classifier(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit,dim=-1)
        softmax_logit = softmax_logit.squeeze()

        max_index = torch.argmax(softmax_logit).item()
        # max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

        category = labelencoder.inverse_transform([max_index])
        return category

    def reply_to_msg(self, msg):
        pass

    

    