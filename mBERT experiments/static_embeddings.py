from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model     = BertModel.from_pretrained('bert-base-multilingual-cased')

# 1. Get the token id for the character
token_id = tokenizer.convert_tokens_to_ids('阔')   # a single character

# 2a. Use the embedding layer directly
static_emb = model.embeddings.word_embeddings.weight[token_id]   # shape (768,)

# 2b. Or average contextual outputs across a corpus
def mean_contextual(token, texts):
    all_vecs = []
    for txt in texts:
        enc = tokenizer(txt, return_tensors='pt')
        out = model(**enc).last_hidden_state[0, tokenizer.convert_tokens_to_ids(token)]
        all_vecs.append(out.detach().cpu())
    return torch.stack(all_vecs).mean(0)

# Example
texts = ["阔的海", "阔的车", "阔的阔"]   # many contexts
static_from_avg = mean_contextual('阔', texts)
