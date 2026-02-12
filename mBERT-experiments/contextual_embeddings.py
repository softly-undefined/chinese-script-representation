from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model      = BertModel.from_pretrained('bert-base-multilingual-cased')

sentence   = "我愛雲朵"          # target character: "雲"
target_char = "雲"

# 1. Tokenize
tokens = tokenizer.tokenize(sentence)          # ['我', '愛', '雲', '朵']
idx    = tokens.index(target_char)            # index of "雲"

# 2. Encode and run the model
inputs  = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)                      # outputs.last_hidden_state

# 3. Extract the contextual embedding for the target character
embedding = outputs.last_hidden_state[0, idx]   # (768,)
