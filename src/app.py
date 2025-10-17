import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform (same as training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load trained model ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/best_mask_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Face Mask Detector 😷", layout="centered")
st.title("🧠 Face Mask Detection App")
st.write("Upload a face image to check if the person is wearing a mask or not.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        img_tensor = transform(image).unsqueeze(0).to(device)
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        labels = ["With Mask 😷", "Without Mask ❌"]
        prediction = labels[preds.item()]

        st.success(f"Prediction: **{prediction}**")
