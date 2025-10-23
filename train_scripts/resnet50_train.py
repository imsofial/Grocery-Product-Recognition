import os
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

DATA_DIRECTORY = "/content/dataset_prepared/dataset_prepared"
train_directory = DATA_DIRECTORY + "/train"
val_directory = DATA_DIRECTORY + "/val"
save_path = '/content/drive/MyDrive/finetuned_models/resnet50_fruits.pth'

os.makedirs(os.path.dirname(save_path), exist_ok=True)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

label2id = {
    'apple/fresh': 0, 'apple/rotten': 1,
    'banana/fresh': 2, 'banana/rotten': 3,
    'orange/fresh': 4, 'orange/rotten': 5,
    'potato/fresh': 6, 'potato/rotten': 7,
    'tomato/fresh': 8, 'tomato/rotten': 9
}

class FruitConditionDataset(Dataset):
    def __init__(self, root_dir, transform=None, label2id=None):
        self.samples = []
        self.transform = transform

        for fruit in os.listdir(root_dir):
            fruit_path = os.path.join(root_dir, fruit)
            if not os.path.isdir(fruit_path):
                continue
            for condition in os.listdir(fruit_path):
                condition_path = os.path.join(fruit_path, condition)
                if not os.path.isdir(condition_path):
                    continue
                label_name = f"{fruit}/{condition}"
                label_id = label2id[label_name]
                for img_name in os.listdir(condition_path):
                    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.samples.append((os.path.join(condition_path, img_name), label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = FruitConditionDataset(train_directory, data_transforms['train'], label2id)
val_dataset = FruitConditionDataset(val_directory, data_transforms['val'], label2id)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label2id)

def get_resnet50(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    
    for param in model.parameters():
        param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model.to(device)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(" Model saved:", save_path)

        scheduler.step()

    print(" Training completed. Best Accuracy:", best_acc)
    return model


model = get_resnet50(num_classes=num_classes, pretrained=True)
trained_model = train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4)
