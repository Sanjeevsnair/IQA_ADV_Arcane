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
        image = image.resize((224, 224))  # Resize for consistency
        boxes, _ = mtcnn.detect(image)

        if boxes is None:
            return []

        emotion_results = []
        for box in boxes:
            face = image.crop((box[0], box[1], box[2], box[3]))
            emotion_data = emotion_detector.detect_emotions(np.array(face))
            if emotion_data:
                # Grab the emotions from the first detected face
                emotion_results.append(emotion_data[0])  # Assumes you want the first face's emotions

        return emotion_results
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

def calculate_rule_of_thirds(image):
    np_image = np.array(image)
    height, width = np_image.shape[:2]
    thirds_height = height // 3
    thirds_width = width // 3
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return False
    for box in boxes:
        x_center, y_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if (
            (abs(x_center - thirds_width) < thirds_width // 2 or abs(x_center - 2 * thirds_width) < thirds_width // 2)
            and (abs(y_center - thirds_height) < thirds_height // 2 or abs(y_center - 2 * thirds_height) < thirds_height // 2)
        ):
            return True
    return False

def calculate_vibrancy(image):
    hsv_image = image.convert("HSV")
    np_image = np.array(hsv_image)
    vibrancy = np.mean(np_image[:, :, 1])
    return vibrancy

def detect_symmetry(image):
    np_image = np.array(image.convert("L"))
    left_half = np_image[:, :np_image.shape[1] // 2]
    right_half = np.flip(np_image[:, np_image.shape[1] // 2:], axis=1)
    symmetry_score = np.sum(left_half == right_half) / left_half.size
    return symmetry_score

def calculate_color_balance(image):
    r, g, b = image.split()
    avg_r = np.mean(np.array(r))
    avg_g = np.mean(np.array(g))
    avg_b = np.mean(np.array(b))
    balance = abs(avg_r - avg_g) + abs(avg_g - avg_b) + abs(avg_b - avg_r)
    return balance

def calculate_exposure(image):
    grayscale_image = image.convert("L")
    np_image = np.array(grayscale_image)
    exposure = np.mean(np_image)
    return exposure

def detect_depth_of_field(image):
    np_image = np.array(image.convert("L"))
    sharpness = np.std(np_image)
    return sharpness > 50

def detect_natural_framing(image):
    edges = image.filter(ImageFilter.FIND_EDGES)
    edge_data = np.array(edges)
    corners = edge_data[:50, :50].sum() + edge_data[-50:, :50].sum() + edge_data[:50, -50:].sum() + edge_data[-50:, -50:].sum()
    return corners > 5000

 
st.title("Image Quality Assessment")
st.write("Upload a folder of images")

 
directory = st.text_input("Enter the folder path containing images:", "")

 
st.sidebar.header("Select Criteria to Apply")
apply_brightness = st.sidebar.checkbox("Apply Brightness", False)
apply_contrast = st.sidebar.checkbox("Apply Contrast", False)
apply_sharpness = st.sidebar.checkbox("Apply Sharpness", False)
apply_saturation = st.sidebar.checkbox("Apply Saturation", False)
apply_temperature = st.sidebar.checkbox("Apply Temperature", False)
apply_highlights_shadows = st.sidebar.checkbox("Apply Highlights & Shadows", False)
apply_rule_of_thirds = st.sidebar.checkbox("Apply Rule of Thirds", False)
apply_vibrancy = st.sidebar.checkbox("Apply Vibrancy", False)
apply_symmetry = st.sidebar.checkbox("Apply Symmetry Detection", False)
apply_color_balance = st.sidebar.checkbox("Apply Color Balance", False)
apply_exposure = st.sidebar.checkbox("Apply Exposure", False)
apply_depth_of_field = st.sidebar.checkbox("Detect Depth of Field", False)
apply_natural_framing = st.sidebar.checkbox("Detect Natural Framing", False)

 
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
if apply_rule_of_thirds:
    rule_of_thirds_required = st.sidebar.checkbox("Require Rule of Thirds", True)
if apply_vibrancy:
    vibrancy_threshold = st.sidebar.slider("Minimum Vibrancy", 0, 255, 100)
if apply_symmetry:
    symmetry_threshold = st.sidebar.slider("Minimum Symmetry Score", 0.0, 1.0, 0.7)
if apply_color_balance:
    color_balance_threshold = st.sidebar.slider("Maximum Color Balance Deviation", 0, 100, 30)
if apply_exposure:
    min_exposure = st.sidebar.slider("Minimum Exposure", 0, 255, 50)
    max_exposure = st.sidebar.slider("Maximum Exposure", 0, 255, 200)
if apply_depth_of_field:
    depth_of_field_required = st.sidebar.checkbox("Require Depth of Field", True)
if apply_natural_framing:
    natural_framing_required = st.sidebar.checkbox("Require Natural Framing", True)

 
count_faces_checkbox = st.sidebar.checkbox("Filter by Face Count", False)
if count_faces_checkbox:
    face_count = st.sidebar.number_input("Select Number of Faces in Image", min_value=1, value=1)
else:
    face_count = 0

 
enable_emotion_detection = st.sidebar.checkbox("Enable Emotion Detection", False)

 
if enable_emotion_detection:
    st.sidebar.header("Select Emotions to Detect")
    emotions_to_detect = st.sidebar.multiselect(
        "Choose Emotions",
        ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
        default=["happy", "sad"],
    )
else:
    emotions_to_detect = []
    
 
enable_aspect_ratio_optimizer = st.sidebar.checkbox("Enable Aspect Ratio Optimization", False)

 
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
enable_duplicate_detection = st.sidebar.checkbox("Enable Duplicate Detection", False)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 1, 20, 10)

 
if (st.sidebar.button("Apply Criteria")):
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
                rule_of_thirds = calculate_rule_of_thirds(image) if apply_rule_of_thirds else None
                vibrancy = calculate_vibrancy(image) if apply_vibrancy else None
                symmetry_score = detect_symmetry(image) if apply_symmetry else None
                color_balance = calculate_color_balance(image) if apply_color_balance else None
                exposure = calculate_exposure(image) if apply_exposure else None
                depth_of_field = detect_depth_of_field(image) if apply_depth_of_field else None
                natural_framing = detect_natural_framing(image) if apply_natural_framing else None
            
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
                    and (not apply_rule_of_thirds or rule_of_thirds == rule_of_thirds_required)
                    and (not apply_vibrancy or vibrancy >= vibrancy_threshold)
                    and (not apply_symmetry or (symmetry_score is not None and symmetry_score >= symmetry_threshold))
                    and (not apply_color_balance or (color_balance is not None and color_balance <= color_balance_threshold))
                    and (not apply_exposure or (exposure is not None and min_exposure <= exposure <= max_exposure))
                    and (not apply_depth_of_field or depth_of_field == depth_of_field_required)
                    and (not apply_natural_framing or natural_framing == natural_framing_required)
                )

             
                emotions_detected = [emotion["emotions"] for emotion in emotion_data if "emotions" in emotion]
                emotion_data = detect_emotions(os.path.join(directory, file)) if enable_emotion_detection else []

                # Check if any detected emotions match the selected emotions
                emotion_matches = any(
                    any(emotion in emotions_to_detect for emotion in detected_emotions)
                    for detected_emotions in emotion_data
                ) if enable_emotion_detection else True

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
                        "rule_of_thirds": rule_of_thirds,
                        "vibrancy": vibrancy,
                        "symmetry_score": symmetry_score,
                        "color_balance": color_balance,
                        "exposure": exposure,
                        "depth_of_field": depth_of_field,
                        "natural_framing": natural_framing,
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
                st.subheader("Images that Meet Criteria")
                for data in image_data:
                    st.image(data["image"], caption=data["file"], use_column_width=True)
                    st.write("Brightness:", data["brightness"])
                    st.write("Contrast:", data["contrast"])
                    st.write("Sharpness:", data["sharpness"])
                    st.write("Saturation:", data["saturation"])
                    st.write("Temperature:", data["temperature"])
                    st.write("Highlights:", data["highlights"])
                    st.write("Shadows:", data["shadows"])
                    st.write("Rule of Thirds:", data["rule_of_thirds"])
                    st.write("Vibrancy:", data["vibrancy"])
                    st.write("Symmetry Score:", data["symmetry_score"])
                    st.write("Color Balance Deviation:", data["color_balance"])
                    st.write("Exposure:", data["exposure"])
                    st.write("Depth of Field:", data["depth_of_field"])
                    st.write("Natural Framing:", data["natural_framing"])
                    st.write("Faces detected:", data["num_faces"])
                    if enable_emotion_detection:
                        st.write("Detected Emotions:", data["emotions"])
                    st.write("---")
            else:
                st.write("No images match the selected criteria.")
        except Exception as e:
            st.error(f"An error occurred: {e}")