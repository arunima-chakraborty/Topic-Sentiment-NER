
import torch
from transformers import DistilBertTokenizerFast
from model_class import MultiTaskModel
from onnxruntime.quantization import quantize_dynamic, QuantType

import pandas as pd
import ast
import torch
from torch.utils.data import Dataset
from torch import nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel, BitsAndBytesConfig,AutoModel
import numpy as np
import logging
import numpy as np
import logging
import contractions
import bitsandbytes as bnb
import contractions
from tqdm import tqdm
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-multilingual-cased")
dummy_text = "Sample text for export"
dummy_inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
input_ids = dummy_inputs["input_ids"].to(device)
attention_mask = dummy_inputs["attention_mask"].to(device)


model = MultiTaskModel()
bin_file = "./multilingual_model.bin"
model.load_state_dict(torch.load(bin_file, map_location=device))
model.to(device)
model.eval()
model.export_mode = True  


onnx_path = "multilingual_model_onnx.onnx"
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["topic_logits", "ner_logits", "sentiment_logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "topic_logits": {0: "batch_size", 1: "seq_len"},
        "ner_logits": {0: "batch_size", 1: "seq_len"},
        "sentiment_logits": {0: "batch_size"}
    },
    opset_version=14
)

quantize_dynamic(onnx_path, onnx_path, weight_type=QuantType.QInt8)

print("ONNX export and quantization complete.")
