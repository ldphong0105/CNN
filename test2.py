import librosa
import librosa.display
import matplotlib.pyplot as plt

# Đọc tệp âm thanh
file_path = "1e9b215e_nohash_1.wav"  # Thay bằng đường dẫn tệp của bạn
data, sr = librosa.load(file_path,sr=22201)  # sr=None để giữ nguyên tần số mẫu gốc

# Vẽ dạng sóng
plt.figure(figsize=(10, 4))
librosa.display.waveshow(data, sr=sr)
plt.title("Waveform of the Audio (Librosa)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()
print(len(data))
