import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_mask_model():
    return load_model("models/mask_detector.keras")


model = load_mask_model()

IMG_SIZE = 128


# =====================================================
# LOAD FACE DETECTOR
# =====================================================

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    st.error("❌ Face detector could not be loaded.")
    st.stop()


# =====================================================
# TITLE
# =====================================================

st.title("😷 Face Mask Detection System")

st.write(
    "Detect whether a person is wearing a face mask "
    "using Machine Learning and Computer Vision."
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Option",
    [
        "Home",
        "Upload Image"
    ]
)


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.header("🏠 Welcome")

    st.markdown("""
    ## Face Mask Detection

    This application uses a trained CNN model to classify
    faces into two categories:

    - 😷 **With Mask**
    - 🚫 **Without Mask**

    ### Technologies Used

    | Technology | Purpose |
    |---|---|
    | Python | Programming |
    | NumPy | Numerical Processing |
    | OpenCV | Face Detection |
    | TensorFlow | Deep Learning |
    | Keras | CNN Model |
    | Streamlit | Web Application |
    """)

    st.success(
        "Select 'Upload Image' from the sidebar to test the model."
    )


# =====================================================
# UPLOAD IMAGE PAGE
# =====================================================

elif page == "Upload Image":

    st.header("🖼️ Face Mask Detection")

    uploaded_file = st.file_uploader(
        "Upload an image containing a face",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")

        frame = np.array(image)

        # Convert RGB → BGR for OpenCV
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )


        # -------------------------------------------------
        # GRAYSCALE FOR FACE DETECTION
        # -------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # -------------------------------------------------
        # DETECT FACES
        # -------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )


        # -------------------------------------------------
        # NO FACE
        # -------------------------------------------------

        if len(faces) == 0:

            st.warning("⚠️ No face detected.")


        # -------------------------------------------------
        # FACE DETECTED
        # -------------------------------------------------

        else:

            st.success(
                f"✅ {len(faces)} face(s) detected."
            )


            # Process every detected face
            for (x, y, w, h) in faces:

                # -------------------------------------------------
                # CROP FACE
                # -------------------------------------------------

                face = frame[y:y+h, x:x+w]

                if face.size == 0:
                    continue


                # -------------------------------------------------
                # RESIZE
                # -------------------------------------------------

                face = cv2.resize(
                    face,
                    (IMG_SIZE, IMG_SIZE)
                )


                # -------------------------------------------------
                # IMPORTANT:
                # OpenCV uses BGR.
                # Keras training images use RGB.
                # Therefore convert BGR → RGB.
                # -------------------------------------------------

                face = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )


                # -------------------------------------------------
                # NORMALIZATION
                # -------------------------------------------------

                face = face.astype(
                    "float32"
                ) / 255.0


                # -------------------------------------------------
                # ADD BATCH DIMENSION
                # -------------------------------------------------

                face = np.expand_dims(
                    face,
                    axis=0
                )


                # -------------------------------------------------
                # MODEL PREDICTION
                # -------------------------------------------------

                prediction = model.predict(
                    face,
                    verbose=0
                )[0][0]


                # -------------------------------------------------
                # CLASSIFICATION
                # -------------------------------------------------

                if prediction >= 0.5:

                    label = "WITHOUT MASK"
                    confidence = prediction * 100

                else:

                    label = "WITH MASK"
                    confidence = (1 - prediction) * 100


                # -------------------------------------------------
                # DRAW FACE RECTANGLE
                # -------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )


                # -------------------------------------------------
                # DRAW LABEL
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}%",
                    (x, max(y - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            # -------------------------------------------------
            # CONVERT BGR → RGB FOR STREAMLIT
            # -------------------------------------------------

            result = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------

            st.image(
                result,
                caption="Detection Result",
                use_container_width=True
            )