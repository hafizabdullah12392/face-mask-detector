import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
def load_model(model_path="models/best_mask_model.pth"):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict function
def predict_image(image_path):
    model = load_model()
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(img_tensor)
    _, preds = torch.max(outputs, 1)
    labels = ["With Mask 😷", "Without Mask ❌"]
    prediction = labels[preds.item()]
    print(f"Prediction: {prediction}")
    return prediction

if __name__ == "__main__":
    # test with any image
    test_image = "data/raw/with_mask/with_mask_1.jpg"  # change path
    predict_image(test_image)
