from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
import io

from model import BinaryClassificationModel


# Завантажуємо модель (потрібно замінити шлях на вашу модель)
#def load_model():
#    model = torch.load('dolphin_binary_classification.pth', map_location=torch.device('cpu'))  # Використовуємо CPU, якщо немає GPU
#    model.eval()  # Перехід моделі в режим оцінки
#    return model

def load_model(model_path, device):
    model = BinaryClassificationModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

model_path = "dolphin_binary_classification.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)

# Ініціалізація Flask
app = Flask(__name__)

# Трансформації для зображень (перетворення в tensor, нормалізація тощо)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')  # Це відобразить HTML сторінку з формою завантаження

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Image processing
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")  # Convert to RGB
        img = transform(img).unsqueeze(0).to(device)  # Apply transform and add batch dimension
    except Exception as e:
        return jsonify({'error': f"Image processing error: {str(e)}"})

    # Prediction
    try:
        with torch.no_grad():
            outputs = model(img)  # Get model output
            confidence = outputs.item()  # Assuming a single scalar output
            prediction = 1 if confidence > 0.5 else 0  # Binary classification
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"})

    # Result mapping
    class_name = "Yes, this dolphin is special!" if prediction == 1 else "No, this dolphin is not special."
    return jsonify({'result': class_name, 'confidence': confidence})


if __name__ == '__main__':
    app.run(debug=True)
