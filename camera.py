import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = 128

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model("models/mask_detector.keras")

print("Model loaded successfully!")

# ==========================================
# LOAD FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("ERROR: Face detector could not be loaded.")
    exit()

print("Face detector loaded successfully!")

# ==========================================
# OPEN WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

# ==========================================
# REAL-TIME DETECTION
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read webcam.")
        break

    # Flip webcam image
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ======================================
    # PROCESS EACH FACE
    # ======================================

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # Resize
        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE)
        )

        # BGR → RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Normalize
        face = face.astype(
            "float32"
        ) / 255.0

        # Add batch dimension
        face = np.expand_dims(
            face,
            axis=0
        )

        # ==================================
        # PREDICTION
        # ==================================

        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        # ==================================
        # CLASSIFICATION
        # ==================================

        # with_mask = 0
        # without_mask = 1

        if prediction >= 0.5:

            label = "WITHOUT MASK"
            confidence = prediction * 100

        else:

            label = "WITH MASK"
            confidence = (1 - prediction) * 100

        # ==================================
        # DRAW RESULT
        # ==================================

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"{label} {confidence:.1f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Face Mask Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# CLOSE CAMERA
# ==========================================

camera.release()
cv2.destroyAllWindows()

print("Webcam stopped.")