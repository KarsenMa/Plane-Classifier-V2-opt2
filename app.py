import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import cv2
import os
import numpy as np
import time

# Configure Streamlit page
st.set_page_config(page_title="Plane Classifier", page_icon="✈️", layout="centered")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Replace with your model
    return model

model = load_model()

# Custom class names
CLASS_NAMES = {
    0: "A300", 1: "A310", 2: "A318", 3: "A319", 4: "A320", 5: "A321",
    6: "B737-2", 7: "B737-3", 8: "B737-4", 9: "B737-5",
    10: "B737-6", 11: "B737-7", 12: "B737-8", 13: "B737-9",
    14: "B737-8 MAX", 15: "B737-9 MAX", 16: "B707", 17: "B727",
    18: "B747", 19: "B757", 20: "B767", 21: "B777", 22: "B787"
}

# Classify and annotate a single image
def classify_image(image):
    results = model.predict(image)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return "No plane detected", 0.0, image

    cls_id = int(boxes.cls[0].item())
    confidence = float(boxes.conf[0].item())
    label = CLASS_NAMES.get(cls_id, f"Unknown ({cls_id})")

    # Draw bounding box
    box = boxes.xyxy[0].tolist()
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=5)
    draw.text((box[0], box[1] - 10), f"{label} {confidence*100:.1f}%", fill="red")

    return label, confidence, image

# Process video frame by frame and stream frames (with frame skipping)
def stream_video(video_path, frame_skip=1):
    cap = cv2.VideoCapture(video_path)

    frames = []
    labels = []
    frame_display = st.empty()
    label_display = st.empty()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps

    progress = st.progress(0)
    frame_idx = 0
    processed_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every N-th frame
        if frame_idx % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Predict
            results = model.predict(pil_img)
            boxes = results[0].boxes

            label_text = "No plane detected"
            draw = ImageDraw.Draw(pil_img)
            if boxes is not None and len(boxes) > 0:
                cls_id = int(boxes.cls[0].item())
                confidence = float(boxes.conf[0].item())
                label_text = f"{CLASS_NAMES.get(cls_id, f'Unknown ({cls_id})')} ({confidence*100:.1f}%)"

                box = boxes.xyxy[0].tolist()
                draw.rectangle(box, outline="red", width=5)
                draw.text((box[0], box[1] - 10), label_text, fill="red")

            # Save frame and label
            frames.append(pil_img)
            labels.append(label_text)

            # Stream frame + label
            frame_display.image(pil_img, caption=f"Frame {processed_idx+1}", use_container_width=True)
            label_display.markdown(f"### ✈️ Prediction: **{label_text}**")
            time.sleep(delay * frame_skip)

            processed_idx += 1

        frame_idx += 1
        progress.progress(min(frame_idx / frame_count, 1.0))

    cap.release()
    progress.empty()

    return frames, labels

# --- Streamlit UI ---

st.title("✈️ Plane Classifier")
st.write("Upload an **image** or a **video** to classify planes!")

file_option = st.radio("Select input type:", ["Image", "Video"])

# Initialize session state for video frames
if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = None
    st.session_state.processed_labels = None

if file_option == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner('Classifying image...'):
            label, confidence, output_image = classify_image(image)

        st.image(output_image, caption="Detected Plane", use_container_width=True)
        st.markdown(f"### ✈️ Prediction: **{label}**")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

elif file_option == "Video":
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # 🚨 CLEAR previously processed frames when new video is uploaded
        st.session_state.processed_frames = None
        st.session_state.processed_labels = None

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.markdown("### ⚡ Speed Options")
        frame_skip = st.slider("Frame skip amount (1 = no skip, higher = faster)", 1, 10, 2)

        if st.button("🚀 Process Video"):
            with st.spinner('Processing video... This may take a while.'):
                frames, labels = stream_video(tfile.name, frame_skip=frame_skip)

            st.session_state.processed_frames = frames
            st.session_state.processed_labels = labels
            st.success("✅ Video processing complete!")

    if st.session_state.processed_frames:
        st.markdown("### 📽️ Replay Slideshow")
        slideshow_speed = st.slider("Slideshow speed (seconds per frame):", 0.01, 1.0, 0.1)
        
        play = st.button("▶️ Play Slideshow")
        if play:
            slideshow_placeholder = st.empty()
            label_placeholder = st.empty()
            for frame, label_text in zip(st.session_state.processed_frames, st.session_state.processed_labels):
                slideshow_placeholder.image(frame, use_container_width=True)
                label_placeholder.markdown(f"### ✈️ Prediction: **{label_text}**")
                time.sleep(slideshow_speed)
