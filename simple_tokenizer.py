#----------------------------------------------
# Chapter 2: Working with Text Data
# I'm generally going to follow the book, 
# But I'll also go off on my own a little bit.
#----------------------------------------------


# Open the text corpus for reading.
# I choose HP Lovecrafts 'Fungi From Yuggoth' series of poems.
with open("lovecraft.txt", 'r', encoding="utf-8") as file:
    raw_text = file.read()

print("Number of Characters: " + str(len(raw_text)))
# Number of Characters: 23377

from collections import Counter
char_count = Counter(raw_text)
print("Character Counts:\n" + str(char_count))
# Character Counts:
#Counter({' ': 3583, 'e': 2224, 't': 1432, 'a': 1375...})

# We want to split the text into tokens. For this example
# it seems like the book just uses words and special characters,
# so we will split on that to get our tokens. This is a little
# bit better then what they do in the book.

special_chars = [key for key in char_count.keys() if not key.isalpha()]
print("Special Chars:\n" + str(special_chars))
#Special Chars:
#['.', ' ', '\n', '-', ',', '—', ';', '’', '!', ':', '“', '”', '?', '(', ')']

import re
pattern = '|'.join(re.escape(c) for c in special_chars)
preprocessed = re.split(f'({pattern})', raw_text)
preprocessed = [c for c in preprocessed if c] # Remove '' values
print("Total Tokens: " + str(len(preprocessed)))
# Total Tokens: 9200
print("First 10 Tokens:\n" + str(preprocessed[:10]))
#First 10 Tokens:
#['I', '.', ' ', 'The', ' ', 'Book', '\n', '\n', 'The', ' ']

# Now we map our tokens to integers (creating a vocabulary)
vocab_words = sorted(set(preprocessed))
vocab_size = len(vocab_words)
print("Vocab Size: " + str(vocab_size))
# Vocab Size: 1647

# create a dict where the keys are the vocab words 
# and the values are their corresponding int (order alphabetically)
vocab = {token : integer for integer, token in enumerate(vocab_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 20:
        break
# ('\n', 0)... ('Alienation', 18) ('All', 19) ('Always', 20)

# Lets make a class
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} # Reverse tokenization

    def encode(self, text):
        char_count = Counter(text)
        special_chars = [key for key in char_count.keys() if not key.isalpha()]
        pattern = '|'.join(re.escape(c) for c in special_chars)
        preprocessed = re.split(f'({pattern})', text)
        preprocessed = [c for c in preprocessed if c] # Remove '' values
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, text):
        text = "".join([self.int_to_str[i] for i in text])
        return text
    
tokenizer = SimpleTokenizerV1(vocab)
text = """Out in the mindless void the daemon bore me,
Past the bright clusters of dimensioned space,
Till neither time nor matter stretched before me,
But only Chaos, without form or place."""

ids = tokenizer.encode(text)
print(ids)
# [164, 1, 876, 1, 1430, 1... 7]
print(tokenizer.decode(ids)) 
# Out in the mindless void the daemon bore me...

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} # Reverse tokenization

    def encode(self, text):
        char_count = Counter(text)
        special_chars = [key for key in char_count.keys() if not key.isalpha()]
        special_chars = [item for item in special_chars if item not in ['<', '>', '|']] # Need to remove these special chars now
        pattern = '|'.join(re.escape(c) for c in special_chars)
        preprocessed = re.split(f'({pattern})', text)
        preprocessed = [c for c in preprocessed if c] # Remove '' values
        preprocessed = [item if item in self.str_to_int else "<|unk|>" 
                        for item in preprocessed] # Add processing for unknown words
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, text):
        text = "".join([self.int_to_str[i] for i in text])
        return text

vocab_words.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token : integer for integer, token in enumerate(vocab_words)}
print(vocab["<|endoftext|>"])
text1 = "Out in the mindless void the daemon bore me"
text2 = "I heard all things in the heaven and in the earth. I heard many things in hell."
text = " <|endoftext|> ".join((text1, text2))
print(text)
# Out in the mindless void the daemon bore me <|endoftext|> I heard all things in the heaven and in the earth. I heard many things in hell.

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
# [164, 1, 876, 1, 1430, 1, 1016, 1, 1550...]
print(tokenizer.decode(tokenizer.encode(text)))
# Out in the mindless void the daemon bore me <|endoftext|> I heard all things in the heaven and in the earth. I heard <|unk|> things in hell.