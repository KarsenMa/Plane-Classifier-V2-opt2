import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw

# Configure Streamlit page
st.set_page_config(page_title="Plane Classifier", page_icon="✈️", layout="centered")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Replace with your new model path if needed
    return model

model = load_model()

# Custom class name mapping
CLASS_NAMES = {
    0: "A300", 1: "A310", 2: "A318", 3: "A319", 4: "A320", 5: "A321",
    6: "B737-2", 7: "B737-3", 8: "B737-4", 9: "B737-5",
    10: "B737-6", 11: "B737-7", 12: "B737-8", 13: "B737-9",
    14: "B737-8 MAX", 15: "B737-9 MAX", 16: "B707", 17: "B727",
    18: "B747", 19: "B757", 20: "B767", 21: "B777", 22: "B787"
}

# Function to classify image and draw bounding box
def classify_image(image):
    results = model.predict(image)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return "No plane detected", 0.0, image  # Return original if no detection

    cls_id = int(boxes.cls[0].item())
    confidence = float(boxes.conf[0].item())
    label = CLASS_NAMES.get(cls_id, f"Unknown ({cls_id})")  # Fallback if ID not mapped

    # Draw bounding box
    box = boxes.xyxy[0].tolist()  # (x1, y1, x2, y2)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=5)
    draw.text((box[0], box[1] - 10), f"{label} {confidence*100:.1f}%", fill="red")

    return label, confidence, image

# Streamlit UI
st.title("✈️ Plane Classifier")
st.write("Upload an image of a plane and see what model it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner('Classifying...'):
        label, confidence, output_image = classify_image(image)

    st.image(output_image, caption="Detected Plane", use_container_width=True)

    st.markdown(f"### ✈️ Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

