import torch

from transformers import LongformerModel, LongformerTokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

input_text = "Your input text goes here."
tokenized_input = tokenizer.encode(input_text, return_tensors='pt')

