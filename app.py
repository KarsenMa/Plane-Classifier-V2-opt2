import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import cv2
import os
import numpy as np

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

# Process video frame by frame
def process_video(video_path):
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame (BGR to RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Classify
        results = model.predict(pil_img)
        boxes = results[0].boxes

        # Draw boxes
        draw = ImageDraw.Draw(pil_img)
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes.cls)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                label = CLASS_NAMES.get(cls_id, f"Unknown ({cls_id})")
                box = boxes.xyxy[i].tolist()
                draw.rectangle(box, outline="red", width=5)
                draw.text((box[0], box[1] - 10), f"{label} {confidence*100:.1f}%", fill="red")

        # Convert back to BGR for OpenCV writing
        output_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(output_frame)

        frame_idx += 1
        progress.progress(min(frame_idx / frame_count, 1.0))

    cap.release()
    out.release()
    progress.empty()
    return temp_output.name

# Streamlit UI
st.title("✈️ Plane Classifier")
st.write("Upload an **image** or a **video** to classify planes!")

file_option = st.radio("Select input type:", ["Image", "Video"])

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
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        with st.spinner('Processing video... This may take a while.'):
            output_video_path = process_video(tfile.name)

        st.video(output_video_path)
