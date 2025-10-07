import os
from geffnet import create_model
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from google.colab import files, drive



DATA_DIRECTORY = "/content/dataset_prepared/dataset_prepared"
train_directory = DATA_DIRECTORY + "/train"
test_directory = DATA_DIRECTORY + "/test"
val_directory = DATA_DIRECTORY + "/val"

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


class FruitConditionDataset(Dataset):
    def __init__(self, root_dir, transform=None, label2id=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

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
                        img_path = os.path.join(condition_path, img_name)
                        self.samples.append((img_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
data_root = test_directory

label2id = {}
counter = 0

for fruit in os.listdir(data_root):
    fruit_path = os.path.join(data_root, fruit)
    if not os.path.isdir(fruit_path):
        continue
    for condition in os.listdir(fruit_path):
        label = f"{fruit}/{condition}"
        label2id[label] = counter
        counter += 1

print(len(label2id))


train_dataset = FruitConditionDataset(
    root_dir=train_directory,
    transform=data_transforms['train'],
    label2id=label2id
)

val_dataset = FruitConditionDataset(
    root_dir=val_directory,
    transform=data_transforms['val'],
    label2id=label2id
)

batch_size = 256 

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label2id)
save_path = '/content/drive/MyDrive/fruit_model/efficientnet_fruits.pth' # where we store model
os.makedirs(os.path.dirname(save_path), exist_ok=True)

def get_model(num_classes, pretrained=True):
    """
    Creates EfficientNet B3 and changes classifier to num_classes
    """
    model = create_model('efficientnet_b3', pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(device)

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
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
        
        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0.0
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
        
        # Saves the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("Model saved:", save_path)
        
        scheduler.step()
    
    print("Train loop is finished. Best Accuracy:", best_acc)
    return model

model = get_model(num_classes=num_classes, pretrained=True)
trained_model = train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4)
