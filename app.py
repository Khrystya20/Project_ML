from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
import io

# Завантажуємо модель (потрібно замінити шлях на вашу модель)
def load_model():
    model = torch.load('path_to_your_model.pth', map_location=torch.device('cpu'))  # Використовуємо CPU, якщо немає GPU
    model.eval()  # Перехід моделі в режим оцінки
    return model

model = load_model()

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

    # Обробка зображення
    try:
        img = Image.open(io.BytesIO(file.read()))
        img = transform(img).unsqueeze(0)  # Перетворення в tensor і додавання batch dimension
    except Exception as e:
        return jsonify({'error': f"Image processing error: {str(e)}"})

    # Передбачення
    try:
        with torch.no_grad():
            outputs = model(img)  # Модель робить передбачення
            prediction = torch.argmax(outputs, dim=1).item()  # отримуємо індекс класу
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"})

    # Повертаємо результат (0 - не особливий, 1 - особливий)
    if prediction == 1:
        return jsonify({'result': 'Yes, this dolphin is special!'})
    else:
        return jsonify({'result': 'No, this dolphin is not special.'})

if __name__ == '__main__':
    app.run(debug=True)
