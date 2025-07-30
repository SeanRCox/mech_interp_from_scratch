# Now we will implement Byte-Pair Encoding (BPE)
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2") # Use the GPT2 Encoding

text1 = "Out in the mindless void the daemon bore me"
text2 = "I heard all things in the heaven and in the earth. I heard many things in hell."
text = " <|endoftext|> ".join((text1, text2))

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# BPE tokenizer breaks down unknown words into known token chunks
# Starts with single character tokens and samples data to build
# longer tokens based on a frequency cutoff.

# Now lets create the input-target pairs using a sliding window.
# First, tokenize the whole text
with open("lovecraft.txt", 'r', encoding="utf-8") as file:
    raw_text = file.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
# 6291

# now create the input-target pairs
context_size = 4
for i in range(1, context_size+1):
    context = enc_text[:i]
    desired = enc_text[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# I ----> .
# I. ---->  The
# I. The ---->  Book
# I. The Book ---->

import torch 
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Now create the dataloader 

def create_dataloader_v1(text, batch_size=4, max_length=256,
                        stride=128, shuffle=True, drop_last=True,
                        num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


