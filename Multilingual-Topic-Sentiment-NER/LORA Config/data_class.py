import pandas as pd
import ast
import torch
from torch.utils.data import Dataset
from torch import nn as nn
from transformers import DistilBertTokenizerFast, TrainingArguments, Trainer, EarlyStoppingCallback, BitsAndBytesConfig,AutoModel, AutoTokenizer,AdamW
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import logging
from sklearn.metrics import accuracy_score as sk_accuracy, f1_score as sk_f1
import numpy as np
import logging
import contractions
import bitsandbytes as bnb


logger = logging.getLogger(__name__)


MAX_LEN = 128
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-multilingual-cased")



class CustomNERTopicDataset(Dataset):
    def __init__(self, df):
        self.texts = []
        self.topics = []
        self.ners = []
        self.sentiments = []
        self.skipped = 0
        self.bio_label_to_id = {
        'O': 0,
        'B': 1,
        'I': 2,
        'E':3,
        'S':4
        
        }

        self.ner_tags_to_id = {
        'O': 0,
        'PER': 1,
        'LOC': 2,
        'ORG': 3,
        'MISC': 4
        }

        for i, row in df.iterrows():
            try:
                text = contractions.fix(row['text'])
                topics = ast.literal_eval(row['topics'])
                ners = ast.literal_eval(row['ner'])
                sentiment = int(row['sentiment'])

                self.texts.append(text)
                self.topics.append(topics)
                self.ners.append(ners)
                self.sentiments.append(sentiment)

            except Exception as e:
                self.skipped += 1
                
        logger.info(f"Dataset initialized. Total skipped rows: {self.skipped}")

    def __len__(self):
        return len(self.texts)
    
    def bio_tagging(self,tokens,topics):
        tokenized_phrase = []
        for topic in topics:
            tokenized_phrase.append(tokenizer.tokenize(topic))

        tags = ['O'] * len(tokens)
        for phrase in tokenized_phrase:
            
            reconstructed_word = ''.join([token.replace('##', '') for token in phrase])
            
            start_index = None
            for i in range(len(tokens)):
                if tokens[i:i+len(phrase)] == phrase:
                    start_index = i
                    break
            
            if start_index is not None:
                
                if len(phrase) == 1:
                    tags[start_index] = 'B'  
                else:
                    tags[start_index] = 'B'  
                    for j in range(1, len(phrase) - 1):
                        tags[start_index + j] = 'I'  
                    tags[start_index + len(phrase) - 1] = 'I'  
        
        
        bio_tags = [self.bio_label_to_id[tag] for tag in tags]
        bio_tags = [-100] + bio_tags
        
        
        if len(bio_tags) > MAX_LEN:
            bio_tags = bio_tags[:MAX_LEN]
        else:
            bio_tags += [-100] * (MAX_LEN - len(bio_tags))
        
        return bio_tags
    
    def ner_tagging(self, tokens, ner):
        output_tags = ['O'] * len(tokens)
        ner_dict = {entity: tag for entity, tag in ner}  

        
        output_tags = ['O'] * len(tokens)

        for i, token in enumerate(tokens):
            
            if token in ner_dict:
                output_tags[i] = ner_dict[token]
            
            elif '##' in token:
                output_tags[i] = 'O'
        
        
        ner_tags = [self.ner_tags_to_id.get(tag, 0) for tag in output_tags]
        ner_tags = [-100] + ner_tags
        if len(ner_tags) > MAX_LEN:
            ner_tags = ner_tags[:MAX_LEN]
        else:
            ner_tags += [-100] * (MAX_LEN - len(ner_tags))
        
        return ner_tags

    def __getitem__(self, idx):
        text = self.texts[idx]
        topic_phrases = self.topics[idx]
        ner_tuples = self.ners[idx]
        sentiment = self.sentiments[idx]

        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        tokens = tokenizer.tokenize(text, truncation=True, max_length=MAX_LEN)

        topic_labels = self.bio_tagging(tokens, topic_phrases)
        ner_labels = self.ner_tagging(tokens, ner_tuples)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "topic_labels": torch.tensor(topic_labels),
            "ner_labels": torch.tensor(ner_labels),
            "sentiment": torch.tensor(sentiment)
        }


