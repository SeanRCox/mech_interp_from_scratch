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



