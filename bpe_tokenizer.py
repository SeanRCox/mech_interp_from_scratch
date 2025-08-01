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

dataloader = create_dataloader_v1(raw_text, 
                                batch_size=1, max_length = 4,
                                stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# [tensor([[  40,   13,  383, 4897]]), tensor([[  13,  383, 4897,  198]])]
# Stride indicates how many positions the input shifts between batches

dataloader = create_dataloader_v1(raw_text, 
                                batch_size=1, max_length = 8,
                                stride=4, shuffle=False)
data_iter = iter(dataloader)
second_batch = next(data_iter)
print(second_batch)
third_batch = next(data_iter)
print(third_batch)
# [tensor([[  40,   13,  383, 4897,  198,  198,  464, 1295]]), tensor([[  13,  383, 4897,  198,  198,  464, 1295,  373]])]
# [tensor([[  198,   198,   464,  1295,   373,  3223,   290, 36972]]), tensor([[  198,   464,  1295,   373,  3223,   290, 36972,   290]])]


# Now lets increase the batches per tensor
dataloader = create_dataloader_v1(raw_text, batch_size=8, 
                                max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


