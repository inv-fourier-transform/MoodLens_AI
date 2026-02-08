from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
# load_dotenv(dotenv_path='../.env')

import os
import streamlit as st

# Get directory where this file (model_helper.py) is located
helper_dir = os.path.dirname(os.path.abspath(__file__))

# Load path from secrets or env
try:
    model_path_relative = st.secrets["MODEL_PATH"]["value"]
except:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(helper_dir, '..', '.env'))
    model_path_relative = os.getenv("MODEL_PATH")

# Construct absolute path (works regardless of working directory)
if os.path.isabs(model_path_relative):
    model_path = model_path_relative
else:
    model_path = os.path.join(helper_dir, model_path_relative)


trained_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


def get_model():
    """Load model directly without custom class."""
    global trained_model

    if trained_model is None:
        # Load ResNet50 directly
        model = models.resnet50(weights=None)

        # Replace FC layer
        model.fc = torch.nn.Linear(model.fc.in_features, 6)

        # Load checkpoint
        # Get path from environment variable
        #model_path = os.getenv("MODEL_PATH") # Refer to .env.example for the path
        model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device)
        model.eval()
        trained_model = model

    return trained_model


def detect_emotion(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    model = get_model()

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)

        probs = probabilities[0].cpu().numpy()
        predicted_idx = probs.argmax()

        return {
            'hard_label': class_labels[predicted_idx],
            'soft_probabilities': {class_labels[i]: float(probs[i]) for i in range(6)},
            'confidence': float(probs[predicted_idx]),
            'all_emotions': class_labels
        }
