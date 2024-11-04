import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import os
from facenet_pytorch import MTCNN
from fer import FER
from imagehash import phash

 
mtcnn = MTCNN(keep_all=True)
emotion_detector = FER()

 
def calculate_brightness(image):
    grayscale_image = image.convert("L")
    np_image = np.array(grayscale_image)
    brightness = np.mean(np_image)
    return brightness

def calculate_contrast(image):
    grayscale_image = image.convert("L")
    np_image = np.array(grayscale_image)
    contrast = np.std(np_image)
    return contrast

def calculate_sharpness(image):
    edge_image = image.filter(ImageFilter.FIND_EDGES)
    np_edge = np.array(edge_image)
    sharpness = np.mean(np_edge)
    return sharpness

def calculate_saturation(image):
    hsv_image = image.convert("HSV")
    np_image = np.array(hsv_image)
    saturation = np.mean(np_image[:, :, 1])
    return saturation

def calculate_temperature(image):
    np_image = np.array(image)
    temperature = np.mean(np_image[:, :, 0]) - np.mean(np_image[:, :, 2])
    return temperature

def calculate_highlights_shadows(image):
    grayscale_image = image.convert("L")
    np_image = np.array(grayscale_image)
    highlights = np.mean(np_image[np_image > 200]) if np_image[np_image > 200].size > 0 else 0
    shadows = np.mean(np_image[np_image < 50]) if np_image[np_image < 50].size > 0 else 0
    return highlights, shadows

def count_faces(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        boxes, _ = mtcnn.detect(image)
        num_faces = len(boxes) if boxes is not None else 0
        return num_faces
    except:
        return 0

def detect_emotions(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        boxes, _ = mtcnn.detect(image)

        if boxes is None:
            return []

        emotion_results = []
        for box in boxes:
            face = image.crop((box[0], box[1], box[2], box[3]))
            emotion_data = emotion_detector.detect_emotions(np.array(face))
            emotion_results.append(emotion_data)

        detected_emotions = []
        for face in emotion_results:
            if "emotions" in face:
                emotions = face["emotions"]
                matching_emotions = [emotion for emotion in emotions if emotion in emotions_to_detect]
                detected_emotions.append(matching_emotions)

        return detected_emotions
    except Exception as e:
        st.error(f"Error detecting emotions: {e}")
        return []
    
def calculate_aspect_ratio(image):
    width, height = image.size
    return width / height

 
def find_duplicates(images, threshold=10):
    duplicates = []
    image_hashes = {}

    for image_name, image in images.items():
        img_hash = phash(image)
        
        for existing_name, existing_hash in image_hashes.items():
            if img_hash - existing_hash < threshold:
                duplicates.append((existing_name, image_name))
        
        image_hashes[image_name] = img_hash

    return duplicates

 
st.title("Image Quality Assessment")
st.write("Upload a folder of images")

 
directory = st.text_input("Enter the folder path containing images:", "")

 
st.sidebar.header("Select Criteria to Apply")
apply_brightness = st.sidebar.checkbox("Apply Brightness", True)
apply_contrast = st.sidebar.checkbox("Apply Contrast", True)
apply_sharpness = st.sidebar.checkbox("Apply Sharpness", True)
apply_saturation = st.sidebar.checkbox("Apply Saturation", True)
apply_temperature = st.sidebar.checkbox("Apply Temperature", True)
apply_highlights_shadows = st.sidebar.checkbox("Apply Highlights & Shadows", True)

 
if apply_brightness:
    brightness_threshold = st.sidebar.slider("Minimum Brightness", 0, 255, 100)
if apply_contrast:
    contrast_threshold = st.sidebar.slider("Minimum Contrast", 0, 100, 20)
if apply_sharpness:
    sharpness_threshold = st.sidebar.slider("Minimum Sharpness", 0, 100, 10)
if apply_saturation:
    saturation_threshold = st.sidebar.slider("Minimum Saturation", 0, 255, 100)
if apply_temperature:
    temperature_threshold = st.sidebar.slider("Temperature (Color Balance)", -100, 100, 0)
if apply_highlights_shadows:
    highlights_threshold = st.sidebar.slider("Minimum Highlights", 0, 255, 200)
    shadows_threshold = st.sidebar.slider("Maximum Shadows", 0, 255, 50)

 
count_faces_checkbox = st.sidebar.checkbox("Filter by Face Count", True)
if count_faces_checkbox:
    face_count = st.sidebar.number_input("Select Number of Faces in Image", min_value=1, value=1)
else:
    face_count = 0

 
enable_emotion_detection = st.sidebar.checkbox("Enable Emotion Detection", True)

 
if enable_emotion_detection:
    st.sidebar.header("Select Emotions to Detect")
    emotions_to_detect = st.sidebar.multiselect(
        "Choose Emotions",
        ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
        default=["happy", "sad"],
    )
else:
    emotions_to_detect = []
    
 
enable_aspect_ratio_optimizer = st.sidebar.checkbox("Enable Aspect Ratio Optimization", True)

 
if enable_aspect_ratio_optimizer:
    st.sidebar.header("Aspect Ratio Optimizer")
    aspect_ratio_options = [
        "16:9",
        "4:3",
        "1:1",
        "3:2",
        "5:4",
    ]
    selected_aspect_ratio = st.sidebar.selectbox(
        "Select Aspect Ratio", aspect_ratio_options
    )
else:
    selected_aspect_ratio = None  

 
st.sidebar.header("Duplicate Detection Settings")
enable_duplicate_detection = st.sidebar.checkbox("Enable Duplicate Detection", True)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 1, 20, 10)

 
if directory:
    try:
         
        image_files = [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        images = {file: Image.open(os.path.join(directory, file)) for file in image_files}
        image_data = []

        for file, image in images.items():
             
            brightness = calculate_brightness(image) if apply_brightness else None
            contrast = calculate_contrast(image) if apply_contrast else None
            sharpness = calculate_sharpness(image) if apply_sharpness else None
            saturation = calculate_saturation(image) if apply_saturation else None
            temperature = calculate_temperature(image) if apply_temperature else None
            highlights, shadows = calculate_highlights_shadows(image) if apply_highlights_shadows else (None, None)
            num_faces = count_faces(os.path.join(directory, file)) if count_faces_checkbox else 0
            
            if enable_aspect_ratio_optimizer:
                aspect_ratio = calculate_aspect_ratio(image)
                 
                selected_ratio = float(selected_aspect_ratio.split(":")[0]) / float(selected_aspect_ratio.split(":")[1])
                 
                aspect_ratio_matches = abs(aspect_ratio - selected_ratio) < 0.01
            else:
                aspect_ratio_matches = True

             
            emotion_data = detect_emotions(os.path.join(directory, file)) if enable_emotion_detection else []

             
            criteria_pass = (
                (not apply_brightness or brightness >= brightness_threshold)
                and (not apply_contrast or contrast >= contrast_threshold)
                and (not apply_sharpness or sharpness >= sharpness_threshold)
                and (not apply_saturation or saturation >= saturation_threshold)
                and (not apply_temperature or temperature >= temperature_threshold)
                and (not apply_highlights_shadows or (highlights >= highlights_threshold and shadows <= shadows_threshold))
                and (not count_faces_checkbox or num_faces == face_count)
                and aspect_ratio_matches 
            )

             
            emotions_detected = [emotion["emotions"] for emotion in emotion_data if "emotions" in emotion]
            emotion_matches = (
                (
                    enable_emotion_detection
                    and any(
                        any(emotion in emotions_to_detect for emotion in detected_emotions)
                        for detected_emotions in emotions_detected
                    )
                )
                if enable_emotion_detection
                else True
            )

            if criteria_pass and emotion_matches:
                image_data.append({
                    "file": file,
                    "image": image,
                    "brightness": brightness,
                    "contrast": contrast,
                    "sharpness": sharpness,
                    "saturation": saturation,
                    "temperature": temperature,
                    "highlights": highlights,
                    "shadows": shadows,
                    "num_faces": num_faces,
                    "emotions": emotions_detected,
                })

         
        if enable_duplicate_detection:
            duplicate_pairs = find_duplicates(images, similarity_threshold)
            if duplicate_pairs:
                st.subheader("Detected Duplicates")
                for img1, img2 in duplicate_pairs:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(images[img1], caption=img1, use_column_width=True)
                    with col2:
                        st.image(images[img2], caption=img2, use_column_width=True)
                    
                    if st.button(f"Delete {img2}", key=img2):
                        os.remove(os.path.join(directory, img2))
                        st.write(f"{img2} has been removed.")
            else:
                st.write("No duplicates found within the specified similarity threshold.")

         
        if image_data:
            st.subheader("Images that Meet Quality and Emotion Criteria")
            for data in image_data:
                st.image(data["image"], caption=data["file"], use_column_width=True)
                st.write("Brightness:", data["brightness"])
                st.write("Contrast:", data["contrast"])
                st.write("Sharpness:", data["sharpness"])
                st.write("Saturation:", data["saturation"])
                st.write("Temperature:", data["temperature"])
                st.write("Highlights:", data["highlights"])
                st.write("Shadows:", data["shadows"])
                st.write("Faces detected:", data["num_faces"])
                if enable_emotion_detection:
                    st.write("Detected Emotions:", data["emotions"])
                st.write("---")
        else:
            st.write("No images match the selected criteria.")
    except Exception as e:
        st.error(f"An error occurred: {e}")