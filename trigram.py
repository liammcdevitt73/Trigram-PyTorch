# Liam McDevitt
# Date: 2023 - 03 - 22

# Training a trigram language model
# Takes in two characters and predicts the third

# E01 of Andrej Karpathy's makemore video
# https://www.youtube.com/watch?v=PaCmpygFfXo

# Imports
import torch
import torch.nn.functional as F

# Load in names from a file
names = open('names.txt', 'r').read().splitlines()

# String to integer & integer to string
chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
combos = [f'{c1}{c2}' for c1 in chars + ['.'] for c2 in chars + ['.']]
stoi2 = {combos[i]:i + 1 + len(stoi) for i in range(len(combos))}
stoi.update(stoi2)
itos = {i:s for s,i in stoi.items()}

# Create the training set of trigrams
xs, ys = [], []
for name in names:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[f'{ch1}{ch2}']
        ix2 = stoi[ch3]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('Number of examples: ', num)

# Initialize the network
g = torch.Generator().manual_seed(0)
W = torch.randn((len(stoi), 27), generator=g, requires_grad=True)

# Training -> Gradient descent
print('\nTRAINING')
print('========')
for k in range(250):

    # Forward pass
    xenc = F.one_hot(xs, num_classes=len(stoi)).float() # Input to the network: one-hot encoding
    logits = xenc @ W # Predict log-counts 
    counts = logits.exp() # Counts
    probs = counts / counts.sum(1, keepdims=True) # Probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
    print(f'Epoch {k+1}: {loss.item()}')

    # Backward pass
    W.grad = None # Set to zero the gradient
    loss.backward()

    # Update
    W.data += -50 * W.grad

# Test / sample from the network
print('\nTESTING')
print('=======')
for i in range(50):

    out = []
    ix = 0

    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=len(stoi)).float() # Input to the network: one-hot encoding
        logits = xenc @ W # Predict log-counts 
        counts = logits.exp() # Counts
        p = counts / counts.sum(1, keepdims=True) # Probabilities for next character

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))