import torch
from transformers import BertForMaskedLM

# ----------------------------
# 1️⃣ Load the trained model
# ----------------------------
model = BertForMaskedLM.from_pretrained("./pepper_dna_model")
model.eval()  # set model to evaluation mode

# ----------------------------
# 2️⃣ Define vocab (same as training)
# ----------------------------
vocab = {
    "[PAD]":0, "[CLS]":1, "[SEP]":2, "[MASK]":3, "[UNK]":4,
    "ATG":5, "TGC":6, "GCG":7, "CGT":8, "GTA":9, "TAC":10
    # Add all k-mers from training here
}
inv_vocab = {v:k for k,v in vocab.items()}
k = 3  # k-mer size

# ----------------------------
# 3️⃣ Prediction function
# ----------------------------
def predict_masked_kmers(sequence, model, vocab, k=3):
    inv_vocab = {v:k for k,v in vocab.items()}
    
    # Tokenize sequence into k-mers
    tokens = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    
    # Encode tokens to ids
    input_ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    input_tensor = torch.tensor([input_ids])
    
    # Run model
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]
    
    # Predict masked tokens
    predictions = tokens.copy()
    for idx, token_id in enumerate(input_ids):
        if token_id == vocab["[MASK]"]:
            pred_id = torch.argmax(logits[0, idx]).item()
            predictions[idx] = inv_vocab[pred_id]
    
    return predictions

# ----------------------------
# 4️⃣ Example usage
# ----------------------------
masked_sequence = "ATG[MASK]ACG"  # Mask the second k-mer
predicted = predict_masked_kmers(masked_sequence, model, vocab, k)
print("Predicted sequence k-mers:", predicted)