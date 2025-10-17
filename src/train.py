import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
from tqdm import tqdm
from data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(epochs=5, lr=0.001):
    train_loader, val_loader = create_dataloaders()

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # freeze base layers

    # Replace last layer for 2 classes (with_mask, without_mask)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {acc:.2f}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/best_mask_model.pth")

    print(f"\n✅ Training complete — Best Validation Accuracy: {best_acc:.2f}%")
    print("✅ Model saved to models/best_mask_model.pth")

if __name__ == "__main__":
    train_model(epochs=5, lr=0.001)
