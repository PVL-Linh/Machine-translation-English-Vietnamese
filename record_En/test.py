# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# import whisper
# import keyboard  # Thêm thư viện này để kiểm tra phím nhấn
#
#
# def record_audio(filename, samplerate=44100, silence_threshold=1000, silence_duration=5):
#     """ Ghi âm từ microphone và lưu thành tệp WAV, ngắt khi không có âm thanh trong 5 giây hoặc khi nhấn 'q' """
#     print("Bắt đầu ghi âm... (Nhấn 'q' để dừng ghi âm)")
#
#     # Tham số
#     chunk_size = int(samplerate * 1)  # 1 giây cho mỗi khung thời gian
#     silence_samples = int(samplerate * silence_duration)  # Tổng số mẫu cho thời gian im lặng
#     recording = []
#     silent_count = 0
#
#     with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
#         print("Đang ghi âm...")
#         while True:
#             # Đọc dữ liệu từ microphone
#             data = stream.read(chunk_size)[0]
#             audio_chunk = np.frombuffer(data, dtype='int16')
#
#             # Kiểm tra mức âm thanh
#             if np.max(np.abs(audio_chunk)) > silence_threshold:
#                 recording.append(data)
#                 silent_count = 0  # Reset silent counter khi phát hiện âm thanh
#             else:
#                 silent_count += len(audio_chunk)
#
#             # Kiểm tra nếu người dùng nhấn 'q' để dừng ghi âm
#             if keyboard.is_pressed('q'):
#                 print("Ngừng ghi âm do nhấn 'q'.")
#                 break
#
#             # Ngừng ghi âm khi không có âm thanh trong thời gian quy định
#             if silent_count >= silence_samples:
#                 print("Ngừng ghi âm do không có âm thanh.")
#                 break
#
#     # Kết hợp tất cả các đoạn ghi âm thành một mảng numpy
#     recording = b''.join(recording)
#     audio_data = np.frombuffer(recording, dtype='int16')
#
#     # Lưu âm thanh với tốc độ mẫu chính xác
#     write(filename, samplerate, audio_data)
#     print(f"Ghi âm đã lưu thành {filename}")
#
#
# def transcribe_audio(filename):
#     # Khởi tạo mô hình Whisper
#     model = whisper.load_model("medium", device="cuda")  # Hoặc "cpu" nếu không có CUDA
#     """ Phiên âm tệp âm thanh sử dụng mô hình Whisper """
#     print(f"Đang phiên âm {filename}")
#     result = model.transcribe(filename, language="en")
#     return result
#
#
# # Ví dụ sử dụng
# def spechToText():
#
#     wav_file = './record_En/audio.wav'
#
#     # Ghi âm và phiên âm
#     record_audio(wav_file)
#     result = transcribe_audio(wav_file)
#     print("Kết quả phiên âm:", result["text"])
#     return str(result["text"])
#


# import sounddevice as sd
# from scipy.io.wavfile import write
# import whisper

# # Khởi tạo mô hình Whisper
# model = whisper.load_model("medium", device="cuda")  # Hoặc "cpu" nếu không có CUDA

# def record_audio(filename, duration=20, samplerate=44100):
#     """ Ghi âm từ microphone và lưu thành tệp WAV """
#     print("Bắt đầu ghi âm...")
#     recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
#     sd.wait()  # Chờ đến khi ghi âm xong
#     write(filename, samplerate, recording)  # Lưu thành WAV
#     print(f"Ghi âm đã lưu thành {filename}")

# def transcribe_audio(filename):
#     """ Phiên âm tệp âm thanh sử dụng mô hình Whisper """
#     print(f"Đang phiên âm {filename}")
#     result = model.transcribe(filename, language="en")
#     return result

# # Ví dụ sử dụng
# wav_file = './record_En/audio.wav'

# # Ghi âm và phiên âm
# record_audio(wav_file)
# result = transcribe_audio(wav_file)
# print("Kết quả phiên âm:", result["text"])


# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# import whisper

# # Khởi tạo mô hình Whisper
# model = whisper.load_model("medium", device="cuda")  # Hoặc "cpu" nếu không có CUDA

# def record_audio(filename, samplerate=44100, silence_threshold=1000, silence_duration=5):
#     """ Ghi âm từ microphone và lưu thành tệp WAV, ngắt khi không có âm thanh trong 5 giây """
#     print("Bắt đầu ghi âm...")

#     # Tham số
#     chunk_size = int(samplerate * 1)  # 0.1 giây cho mỗi khung thời gian
#     silence_samples = int(samplerate * silence_duration)  # Tổng số mẫu cho thời gian im lặng
#     recording = []
#     silent_count = 0

#     with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
#         print("Đang ghi âm...")
#         while True:
#             # Đọc dữ liệu từ microphone
#             data = stream.read(chunk_size)[0]
#             audio_chunk = np.frombuffer(data, dtype='int16')

#             # Kiểm tra mức âm thanh
#             if np.max(np.abs(audio_chunk)) > silence_threshold:
#                 recording.append(data)
#                 silent_count = 0  # Reset silent counter khi phát hiện âm thanh
#             else:
#                 silent_count += len(audio_chunk)

#             # Ngừng ghi âm khi không có âm thanh trong thời gian quy định
#             if silent_count >= silence_samples:
#                 print("Ngừng ghi âm do không có âm thanh.")
#                 break

#     # Kết hợp tất cả các đoạn ghi âm thành một mảng numpy
#     recording = b''.join(recording)
#     audio_data = np.frombuffer(recording, dtype='int16')

#     # Lưu âm thanh với tốc độ mẫu chính xác
#     write(filename, samplerate, audio_data)
#     print(f"Ghi âm đã lưu thành {filename}")

# def transcribe_audio(filename):
#     """ Phiên âm tệp âm thanh sử dụng mô hình Whisper """
#     print(f"Đang phiên âm {filename}")
#     result = model.transcribe(filename, language="en")
#     return result

# # Ví dụ sử dụng
# wav_file = './record_En/audio.wav'

# # Ghi âm và phiên âm
# record_audio(wav_file)
# result = transcribe_audio(wav_file)
# print("Kết quả phiên âm:", result["text"])
