from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch


model_name = 'agro-nucleotide-transformer-1b'

# fetch model and tokenizer from InstaDeep's hf repo
agro_nt_model = AutoModelForMaskedLM.from_pretrained(f'InstaDeepAI/{model_name}')
agro_nt_tokenizer = AutoTokenizer.from_pretrained(f'InstaDeepAI/{model_name}')

print(f"Loaded the {model_name} model with {agro_nt_model.num_parameters()} parameters and corresponding tokenizer.")

# example sequence and tokenization
sequences = ['ATATACGGCCGNC','GGGTATCGCTTCCGAC']

batch_tokens = agro_nt_tokenizer(sequences,padding="longest")['input_ids']
print(f"Tokenzied sequence: {agro_nt_tokenizer.batch_decode(batch_tokens)}")

torch_batch_tokens = torch.tensor(batch_tokens)
attention_mask = torch_batch_tokens != agro_nt_tokenizer.pad_token_id

# inference
outs = agro_nt_model(
    torch_batch_tokens,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True
)

# get the final layer embeddings and language model head logits
embeddings = outs['hidden_states'][-1].detach().numpy()
logits = outs['logits'].detach().numpy()
predicted_ids = torch.argmax(torch.tensor(logits), dim=-1)  # pick highest score at each position

# Convert IDs back to tokens
predicted_sequences = [agro_nt_tokenizer.convert_ids_to_tokens(ids) for ids in predicted_ids]

# Convert token list to string
predicted_sequences_str = ["".join(seq) for seq in predicted_sequences]

print("\nPredicted sequences:")
for i, seq in enumerate(predicted_sequences_str):
    print(f"Original:  {sequences[i]}")
    print(f"Predicted: {seq}")

