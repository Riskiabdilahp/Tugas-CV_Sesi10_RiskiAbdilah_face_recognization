import cv2

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat mengakses kamera.")
        break
    
    # Cerminkan tampilan (horizontal flip)
    mirrored_frame = cv2.flip(frame, 1)

    # Tampilkan hasil mirror
    cv2.imshow("Kamera Mirror", mirrored_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
