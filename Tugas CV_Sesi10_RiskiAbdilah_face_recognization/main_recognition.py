import cv2
from deepface import DeepFace
import os

# Pastikan file Haar Cascade, model, dan mapping tersedia
HAAR_CASCADE_PATH = r"D:\MAPEL NUSA PUTRA\SEMESTER5\Computer Vision\New folder (10)\Tugas CV_Sesi10_RiskiAbdilah_face_recognization\haarcascade_frontalface_default.xml"
TRAINER_PATH = r"D:\MAPEL NUSA PUTRA\SEMESTER5\Computer Vision\New folder (10)\Tugas CV_Sesi10_RiskiAbdilah_face_recognization\trainer.yml"
NAME_MAPPING_PATH = r"D:\MAPEL NUSA PUTRA\SEMESTER5\Computer Vision\New folder (10)\Tugas CV_Sesi10_RiskiAbdilah_face_recognization\name_mapping.txt"

if not os.path.exists(HAAR_CASCADE_PATH):
    print(f"Error: File Haar Cascade tidak ditemukan di {HAAR_CASCADE_PATH}.")
    exit()

if not os.path.exists(TRAINER_PATH):
    print(f"Error: File model pengenalan wajah (trainer.yml) tidak ditemukan di {TRAINER_PATH}.")
    exit()

if not os.path.exists(NAME_MAPPING_PATH):
    print(f"Error: File mapping nama (name_mapping.txt) tidak ditemukan di {NAME_MAPPING_PATH}.")
    exit()

# Load Haar Cascade dan model pengenalan wajah
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)

# Load mapping nama
name_mapping = {}
with open(NAME_MAPPING_PATH, 'r') as f:
    for line in f.readlines():
        name, idx = line.strip().split(":")
        name_mapping[int(idx)] = name

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat mengakses webcam.")
    exit()

print("Tekan 'q' untuk keluar dari program.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari webcam.")
        break

    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Prediksi ID wajah
        try:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 80:  # Threshold confidence
                name = name_mapping.get(id, "Unknown")
            else:
                name = "Unknown"
        except Exception as e:
            print(f"Error during face recognition: {e}")
            name = "Unknown"

        # Deteksi ekspresi wajah menggunakan DeepFace
        face_roi = frame[y:y+h, x:x+w]  # Region of Interest (ROI) untuk wajah
        try:
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            
            # Penanganan format keluaran
            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]
            
            emotion = analysis.get("dominant_emotion", "Unknown") if isinstance(analysis, dict) else "Unknown"
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            emotion = "Unknown"

        # Tampilkan nama dan ekspresi di layar
        text = f"{name} ({emotion})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame di layar
    cv2.imshow('Face Recognition and Emotion Detection', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
