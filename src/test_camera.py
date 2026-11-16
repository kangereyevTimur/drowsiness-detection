
import cv2
import mediapipe as mp

# Загружаем модель определения лиц
detector = mp.tasks.vision.FaceDetector.create_from_options(
    mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path="models/blaze_face_short_range.tflite"
        )
    )
)

# Подключаем камеру MacBook
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

print("Камера запущена. Нажми Q для выхода.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Переводим изображение в RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Преобразуем кадр для MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    # Рисуем рамки вокруг найденных лиц
    for face in result.detections:
        bbox = face.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Drowsiness Detection - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
