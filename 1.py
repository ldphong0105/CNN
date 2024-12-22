import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Đường dẫn tới thư mục chứa dữ liệu train, test, validation
train_data_path = 'train/'
test_data_path = 'test/'
val_data_path = 'validation/'

# Đọc nhãn từ file
def load_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            labels.append([int(i) for i in line.strip().split()])
    return np.array(labels)

# Tải hình ảnh và nhãn tương ứng
def load_data(image_folder, label_file):
    images = []
    labels = load_labels(label_file)
    image_files = sorted(os.listdir(image_folder))
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('L')  # Chuyển về grayscale
            img = img.resize((128, 128))  # Resize ảnh về 128x128
            images.append(np.array(img))
    
    images = np.array(images) / 255.0  # Chuẩn hóa về [0, 1]
    images = images[..., np.newaxis]  # Thêm channel (1) cho grayscale
    return images, labels


# Tải dữ liệu train, test và validation
X_train, y_train = load_data(train_data_path, "train_labels.txt")
X_test, y_test = load_data(test_data_path, "test_labels.txt")
X_val, y_val = load_data(val_data_path, "val_labels.txt")

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 lớp tương ứng với bird, cat, dog
])

# Compile mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)

# Đánh giá mô hình trên tập test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Test Loss: {test_loss:.2f}")

# Vẽ đồ thị kết quả huấn luyện
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
