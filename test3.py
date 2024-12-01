import librosa
import numpy as np
import matplotlib.pyplot as plt

# Đọc tệp âm thanh
file_path = "0a7c2a8d_nohash_0.wav"  # Thay bằng đường dẫn tệp của bạn
data, sr = librosa.load(file_path)

# Đặt ngưỡng biên độ (threshold)
threshold = 0.1  # Tùy chỉnh ngưỡng theo dữ liệu của bạn (thử nghiệm với các giá trị khác nếu cần)

# Tìm vùng tín hiệu lớn hơn ngưỡng
mask = np.abs(data) > threshold
start_index = np.argmax(mask)  # Chỉ số bắt đầu của vùng có tín hiệu lớn
end_index = len(mask) - np.argmax(mask[::-1]) - 1  # Chỉ số kết thúc của vùng có tín hiệu lớn

# Cắt tín hiệu
focused_data = data[start_index:end_index + 1]
focused_time = np.linspace(start_index / sr, end_index / sr, num=len(focused_data))

# Vẽ tín hiệu đã được lọc
plt.figure(figsize=(10, 4))
plt.plot(focused_time, focused_data, label="Focused Signal")
plt.title("Focused Signal (High Amplitude Region)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print(len(focused_data))
