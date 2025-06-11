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


MAX_LEN = 512
model_name = "distilbert/distilbert-base-multilingual-cased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()

        def initialize_weights(self):
            for name, param in self.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

        self.classifiers = {
            "topic_label_map": 5,  
            "ner_label_map": 5,
            "sentiment_label_map": 3
        }

        self.weights = {
            "topic": 0.7,
            "ner": 0.1,
            "sentiment": 0.2
        }

        
        self.bert = AutoModel.from_pretrained(
            model_name,
            load_in_8bit=True,
        )

        
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin", "k_lin"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.bert = get_peft_model(self.bert, self.lora_config).to(torch.float32)

        
        self.dropout = nn.Dropout(0.1)

        
        self.topic_classifier = bnb.nn.Linear8bitLt(self.bert.config.hidden_size, self.classifiers["topic_label_map"]).cuda()
        self.ner_classifier = bnb.nn.Linear8bitLt(self.bert.config.hidden_size, self.classifiers["ner_label_map"]).cuda()
        self.sentiment_classifier = bnb.nn.Linear8bitLt(self.bert.config.hidden_size, self.classifiers["sentiment_label_map"]).cuda()

    def forward(self, input_ids, attention_mask, topic_labels=None, ner_labels=None, sentiment=None):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state).float()
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0]).float()

        
        topic_logits = self.topic_classifier(sequence_output)
        ner_logits = self.ner_classifier(sequence_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)

        loss = None
        if topic_labels is not None and ner_labels is not None and sentiment is not None:
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            topic_loss = loss_fct(topic_logits.view(-1, topic_logits.size(-1)), topic_labels.view(-1))
            ner_loss = loss_fct(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))
            sentiment_loss = loss_fct(sentiment_logits, sentiment)
            
            
            loss = self.weights["topic"] * topic_loss + self.weights["ner"] * ner_loss + self.weights["sentiment"] * sentiment_loss

           
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN DETECTED")
                print("  Topic Loss:", topic_loss.item())
                print("  NER Loss:", ner_loss.item())
                print("  Sentiment Loss:", sentiment_loss.item())
                raise ValueError("Loss became NaN")

        return {
            "loss": loss,
            "topic_logits": topic_logits,
            "ner_logits": ner_logits,
            "sentiment_logits": sentiment_logits,
        }
