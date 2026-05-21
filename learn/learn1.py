
with open("input.txt", "r") as f:
    text = f.read()

print("text length:", len(text))

chars = sorted(list(set(text)))
print(''.join(chars))
vocab_size = len(chars)
print("Vocabulary size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hello"))
print(decode(encode("hello")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

print("training data length:", len(train_data))
print("validation data length:", len(val_data))

block_size = 8
print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")

