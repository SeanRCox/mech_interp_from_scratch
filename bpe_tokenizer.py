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