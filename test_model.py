import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==============================
# LOAD MODEL
# ==============================

model = load_model("models/mask_detector.keras")

IMG_SIZE = 128

# ==============================
# LOAD FACE DETECTOR
# ==============================

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("ERROR: Face detector could not be loaded.")
    exit()

# ==============================
# LOAD TEST IMAGE
# ==============================

frame = cv2.imread("test.jpg")

if frame is None:
    print("ERROR: test.jpg could not be loaded.")
    exit()

# ==============================
# CONVERT TO GRAYSCALE
# ==============================

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)

# ==============================
# DETECT FACE
# ==============================

faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60)
)

print("Faces detected:", len(faces))

# ==============================
# PREDICT
# ==============================

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

    # BGR -> RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Normalize
    face = face.astype("float32") / 255.0

    # Add batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        face,
        verbose=0
    )[0][0]

    print("Prediction:", prediction)

    # Class mapping
    if prediction >= 0.5:

        label = "WITHOUT MASK"
        confidence = prediction * 100

    else:

        label = "WITH MASK"
        confidence = (1 - prediction) * 100

    print("Class:", label)
    print(f"Confidence: {confidence:.2f}%")

    # Draw result
    cv2.rectangle(
        frame,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"{label} {confidence:.2f}%",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# ==============================
# DISPLAY
# ==============================

cv2.imshow(
    "Face Mask Detection Test",
    frame
)

cv2.waitKey(5000)
cv2.destroyAllWindows()