from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch


model_name = 'agro-nucleotide-transformer-1b'

# fetch model and tokenizer from InstaDeep's hf repo
model = AutoModelForMaskedLM.from_pretrained(f'InstaDeepAI/{model_name}')
tokenizer = AutoTokenizer.from_pretrained(f'InstaDeepAI/{model_name}')

model.eval()
sequence = "CTAAACCCTAAACCCTAAACCCTAAACCCTAAACCCCTAACCCTAAACCCTAACCAAAACCCTAAACCCTAAACCCCTAAACCCTAAACCCTAACCTAAACCTACCTAAACCATACATTG"


# Let's say we want to mask the 3rd 6-mer (positions 12–17)
masked_sequence = sequence[:12] + tokenizer.mask_token + sequence[18:]

print("Original sequence:", sequence)
print("Masked sequence:  ", masked_sequence)
inputs = tokenizer(masked_sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

# Get top 5 predicted tokens
topk = torch.topk(logits[0, mask_index], k=5, dim=-1)
predicted_tokens = [tokenizer.decode([idx.item()]) for idx in topk.indices[0]]

print("Top predictions for [MASK]:")
for i, tok in enumerate(predicted_tokens, start=1):
    print(f"{i}. {tok}")