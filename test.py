import librosa
import matplotlib.pyplot as plt
import numpy as np

# Load file âm thanh
y, sr = librosa.load('ffb86d3c_nohash_0.wav')

# Tính toán spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Hiển thị spectrogram
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                          y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency 1  spectrogram')
plt.tight_layout()
plt.show()
# Lấy tần số tại khung thời gian thứ 10
frequencies = librosa.mel_frequencies(n_mels=128)
frame_index = 10
frequency_values = S[:, frame_index]

# In ra các giá trị tần số và cường độ tương ứng
print(frequencies)
#print(frequency_values)