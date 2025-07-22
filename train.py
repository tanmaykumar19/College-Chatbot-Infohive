import json
import torch
import numpy as np
import nltk
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt')

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __len__(self): return len(X_train)
    def __getitem__(self, idx): return X_train[idx], y_train[idx]

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
model = NeuralNet(input_size, hidden_size, output_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    for (words, labels) in train_loader:
        words, labels = words.to(device), labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

print('Training complete. Saving model...')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# Save the trained model
torch.save({
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'all_words': all_words,
    'tags': tags,
    'model_state': model.state_dict()
}, 'data.pth')

print("Training completed and model saved!")

