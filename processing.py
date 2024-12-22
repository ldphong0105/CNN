import os
import glob
import shutil
import random
import gc
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Tạo thư mục
image_data='data/'
train_data_path = 'train/'
val_data_path = 'validation/'
test_data_path = 'test/'
y = []

os.makedirs(train_data_path, exist_ok=True)
os.makedirs(val_data_path, exist_ok=True)
os.makedirs(test_data_path, exist_ok=True)

# Hàm tạo ảnh
def create_image(filename, name, file_path):
    """Convert audio to mel-spectrogram image and save as PNG."""
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    output_path = os.path.join(file_path, f"{name}.png")
    plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S

# Các danh mục và nhãn
categories = {
    'bird': [1, 0, 0],
    'cat': [0, 1, 0],
    'dog': [0, 0, 1],
}

# Tạo ảnh từ file wav và ghi nhãn
base_path = 'Animals/'  
for category, label in categories.items():
    wav_path = os.path.join(base_path, category, '*.wav') 
    for file in glob.glob(wav_path):
        name = os.path.basename(file).split('.')[0]
        create_image(file, name, image_data)
        y.append(label)
    gc.collect()

# Lưu nhãn vào file label.txt
with open("label.txt", "w") as f:
    for label in y:
        f.write(" ".join(map(str, label)) + "\n")

# Đường dẫn tới thư mục chứa file
folder_path = image_data
label_file_path = 'label.txt'

# Đọc danh sách file trong folder
file_list = sorted(os.listdir(folder_path))  # Sắp xếp để khớp thứ tự file với label

# Đọc dữ liệu label từ file label.txt
with open(label_file_path, 'r') as f:
    labels = [line.strip() for line in f]

# Ghép file và label thành cặp
data_pairs = list(zip(file_list, labels))

# Trộn ngẫu nhiên các cặp
random.shuffle(data_pairs)

# Chia thành train (80%), validation (10%), test (10%)
total_files = len(data_pairs)
train_split = int(total_files * 0.8)
val_split = int(total_files * 0.1)

train_data = data_pairs[:train_split]
val_data = data_pairs[train_split:train_split + val_split]
test_data = data_pairs[train_split + val_split:]

# Hàm di chuyển file và ghi nhãn
def move_files_and_save_labels(data, dest_folder, label_file_name):
    os.makedirs(dest_folder, exist_ok=True)
    with open(label_file_name, 'w') as label_file:
        for file_name, label in data:
            src_path = os.path.join(folder_path, file_name)
            dest_path = os.path.join(dest_folder, file_name)
            
            # Kiểm tra và di chuyển file
            if os.path.isfile(src_path):
                shutil.move(src_path, dest_path)
            
            # Ghi nhãn vào file
            label_file.write(f"{label}\n")

# Di chuyển file và tạo file nhãn cho train, validation, test
move_files_and_save_labels(train_data, train_data_path, 'train_labels.txt')
move_files_and_save_labels(val_data, val_data_path, 'val_labels.txt')
move_files_and_save_labels(test_data, test_data_path, 'test_labels.txt')

print("Dữ liệu đã được chia thành train, validation và test.")
print(f"Số lượng file train: {len(train_data)}")
print(f"Số lượng file validation: {len(val_data)}")
print(f"Số lượng file test: {len(test_data)}")
