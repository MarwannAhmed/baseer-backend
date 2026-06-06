import os

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from app.features.text_extraction.config import parameters

SPLIT         = "byclass"
MAX_PER_CLASS = 1000
BATCH_SIZE    = 64
EPOCHS        = 15
LEARNING_RATE = 0.001


class EMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images) / 255.0
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx].unsqueeze(0), self.labels[idx]


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def _load_emnist():
    train = datasets.EMNIST(root=parameters["train_dirs"]["emnist"], split=SPLIT, train=True, download=True)
    test = datasets.EMNIST(root=parameters["train_dirs"]["emnist"], split=SPLIT, train=False, download=True)
    
    imgs = np.concatenate([train.data.numpy(), test.data.numpy()], axis=0)
    labels = np.concatenate([train.targets.numpy(), test.targets.numpy()], axis=0)
    mapping = dict(train.class_to_idx)
    idx_to_char = {v: k for k, v in mapping.items()}
    return imgs, labels, idx_to_char

def _map_to_lowercase(char):
    if char is None:
        return None
    if len(char) == 1 and char.isalpha():
        return char.lower()
    return char

def _prepare_samples(imgs, labels, idx_to_char):
    from collections import defaultdict
    
    buckets = defaultdict(list)
    for img, label in zip(imgs, labels):
        buckets[label].append(img)
    
    samples, targets = [], []
    for label, images in buckets.items():
        char = idx_to_char.get(int(label), None)
        if char is None:
            continue
        
        # char_lower = _map_to_lowercase(char)
        # if char_lower is None:
        #     continue
        
        subset = images[:MAX_PER_CLASS]
        for img in subset:
            arr = img.numpy() if hasattr(img, "numpy") else img
            arr = np.transpose(arr)
            arr = ((1.0 - arr / 255.0) * 255).astype(np.uint8)
            samples.append(arr)
            targets.append(char)
    
    return samples, targets

def train():
    os.makedirs(parameters["train_dirs"]["models"], exist_ok=True)
    os.makedirs(parameters["train_dirs"]["emnist"], exist_ok=True)
    
    imgs, labels, idx_to_char = _load_emnist()
    
    samples, targets = _prepare_samples(imgs, labels, idx_to_char)
    unique_classes = sorted(set(targets))
    print(f"  {len(samples)} samples across {len(unique_classes)} classes")
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(targets)
    num_classes = len(encoder.classes_)
    
    X_train, X_val, y_train, y_val = train_test_split(
        samples, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    
    train_dataset = EMNISTDataset(X_train, y_train)
    val_dataset = EMNISTDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    torch.save(model.state_dict(), parameters["CNN"]["model_path"])
    with open(parameters["CNN"]["encoder_path"], "wb") as f:
        pickle.dump(encoder, f)
    print(f"Saved: {parameters['CNN']['model_path']}, {parameters['CNN']['encoder_path']}")

if __name__ == "__main__":
    train()