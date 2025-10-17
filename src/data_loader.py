import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.images = []
        self.labels = []

        for label, folder in enumerate(["with_mask", "without_mask"]):
            folder_path = os.path.join(data_dir, folder)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(folder_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(data_dir="data/raw", batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = MaskDataset(data_dir, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")
