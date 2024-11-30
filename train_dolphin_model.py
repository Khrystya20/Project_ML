import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

#1. Створення класу для завантаження датасету
class DolphinDataset(Dataset):
    def __init__(self, special_folder, general_folder, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Завантаження "особливих" дельфінів (label=1)
        for img_file in os.listdir(special_folder):
            img_path = os.path.join(special_folder, img_file)
            if os.path.isfile(img_path):
                self.image_paths.append(img_path)
                self.labels.append(1)  # Особливий дельфін

        # Завантаження "всіх" дельфінів (label=0)
        for img_file in os.listdir(general_folder):
            img_path = os.path.join(general_folder, img_file)
            if os.path.isfile(img_path):
                self.image_paths.append(img_path)
                self.labels.append(0)  # Звичайний дельфін

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. Налаштування трансформацій
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Завантаження даних
special_folder = "dataset/special_dolphins"  # Папка з фото особливих дельфінів
general_folder = "dataset/all_dolphins"      # Папка з фото всіх інших дельфінів
full_dataset = DolphinDataset(special_folder, general_folder, transform=transform)

# 4. Розбиття на тренувальні та тестові вибірки
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 5. Створення DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 6. Створення простої нейронної мережі
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Два класи: особливий і звичайний
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 7. Навчання моделі
def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        # Оцінка на тестових даних
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set: {100 * correct / total}%")

# 8. Завантаження нового зображення для передбачення
def predict(model, image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Додаємо вимір для батчу
    output = model(image)
    _, predicted = torch.max(output, 1)
    return "Особливий дельфін" if predicted.item() == 1 else "Звичайний дельфін"

# 9. Головний блок
if __name__ == "__main__":
    model = SimpleCNN()
    train_model(model, train_loader, test_loader, epochs=10)

    # Приклад передбачення для нового зображення
    new_image_path = "new_dolphin.jpg"  # Замініть на шлях до вашого зображення
    result = predict(model, new_image_path)
    print(f"Результат: {result}")
