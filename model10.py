# ===========================
# 1️⃣ Import libraries
# ===========================
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, Trainer, TrainingArguments
from tqdm import tqdm

# ===========================
# 2️⃣ Example DNA sequences
# ===========================
sequences = [
    "ATGCGTACGTTAGCTAGCTAGCTA",
    "GCTAGCTAGGCGTATCGTACGTAG",
    "CGTATGCTAGCTAGCATCGATGCT",
    "TATCGATGCTAGCATCGTAGCTAG",
    "GCTAGCATCGTACGTAGCTAGTCA",
    "ATCGTAGCTAGCATCGTACGTGCA",
    "GCTAGCTAGCTAGCATCGATGCTA",
    "CGTAGCTAGCTAGCATCGTAGCAT",
    "TACGTAGCTAGCTAGCTAGCATCG",
    "GCTAGCATCGTAGCTAGCTAGCTA",
]

# ===========================
# 3️⃣ Tokenization: convert DNA into k-mers
# ===========================
def kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

# Build vocabulary
k = 3
all_kmers = sorted(set(kmer for seq in sequences for kmer in kmers(seq, k)))
vocab = {kmer: i+5 for i, kmer in enumerate(all_kmers)}  # +5 to reserve special tokens
vocab["[PAD]"] = 0
vocab["[CLS]"] = 1
vocab["[SEP]"] = 2
vocab["[MASK]"] = 3
vocab["[UNK]"] = 4

# Reverse vocab for decoding
inv_vocab = {v: k for k, v in vocab.items()}

# ===========================
# 4️⃣ Encode sequences
# ===========================
def encode_sequence(seq):
    tokens = kmers(seq, k)
    return [vocab.get(t, vocab["[UNK]"]) for t in tokens]

encoded_seqs = [encode_sequence(seq) for seq in sequences]

# ===========================
# 5️⃣ Build dataset class
# ===========================
class DNADataset(Dataset):
    def __init__(self, encoded_sequences, mask_prob=0.15):
        self.data = encoded_sequences
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx].copy()
        labels = ids.copy()

        # Randomly mask 15% of tokens
        for i in range(len(ids)):
            if torch.rand(1).item() < self.mask_prob:
                ids[i] = vocab["[MASK]"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

dataset = DNADataset(encoded_seqs)

# ===========================
# 6️⃣ Define a tiny BERT model
# ===========================
config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_position_embeddings=128
)
model = BertForMaskedLM(config)

# ===========================
# 7️⃣ Training setup
# ===========================
training_args = TrainingArguments(
    output_dir="./pepper_dna_model",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=5e-4,
    logging_steps=2,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ===========================
# 8️⃣ Train
# ===========================
trainer.train()

# ===========================
# 9️⃣ Save model
# ===========================
model.save_pretrained("./pepper_dna_model")
print("✅ Model trained and saved!")
