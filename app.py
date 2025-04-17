import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from mtcnn import MTCNN
import tempfile

# Initialize face detector
detector = MTCNN()
detected_faces = []
blurred_faces = set()
original_image = None
blurred_image = None  # Store a version with only blurring (no boxes)

# Function to detect faces
def detect_faces(image):
    global detected_faces, original_image, blurred_faces, blurred_image
    image = np.array(image)
    original_image = image.copy()
    blurred_image = image.copy()  # Store clean version without boxes
    blurred_faces.clear()

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detected_faces = detector.detect_faces(image)

    for idx, face in enumerate(detected_faces):
        x, y, w, h = face['box']
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)

        # Draw bounding box (only for display)
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image_bgr, str(idx + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

# Function to toggle blur on selected face
def toggle_blur_face(image, evt: gr.SelectData, blur_slider_value):
    global detected_faces, original_image, blurred_faces, blurred_image
    if not detected_faces:
        return image  # No faces detected

    image = np.array(image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    click_x, click_y = evt.index[0], evt.index[1]

    for idx, face in enumerate(detected_faces):
        x, y, w, h = face['box']
        if x <= click_x <= x + w and y <= click_y <= y + h:
            face_id = idx

            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

            if face_id in blurred_faces:
                image_bgr[y:y+h, x:x+w] = original_bgr[y:y+h, x:x+w]
                blurred_faces.remove(face_id)
                blurred_image[y:y+h, x:x+w] = original_image[y:y+h, x:x+w]
            else:
                clean_patch = original_bgr[y:y+h, x:x+w]
                blur_kernel = max(3, (blur_slider_value // 2) * 2 + 1)
                blurred_face_bgr = cv2.GaussianBlur(clean_patch, (blur_kernel, blur_kernel), 0)
                image_bgr[y:y+h, x:x+w] = blurred_face_bgr
                blurred_face_rgb = cv2.cvtColor(blurred_face_bgr, cv2.COLOR_BGR2RGB)
                blurred_image[y:y+h, x:x+w] = blurred_face_rgb
                blurred_faces.add(face_id)
            break

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

# Function to blur all faces with current blur slider value
def blur_all_faces(image, blur_slider_value):
    global detected_faces, original_image, blurred_faces, blurred_image

    if not detected_faces:
        return image  # No faces to blur

    image_bgr = cv2.cvtColor(original_image.copy(), cv2.COLOR_RGB2BGR)
    blur_kernel = max(3, (blur_slider_value // 2) * 2 + 1)

    for idx, face in enumerate(detected_faces):
        x, y, w, h = face['box']
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)

        clean_patch = image_bgr[y:y+h, x:x+w]
        blurred_patch = cv2.GaussianBlur(clean_patch, (blur_kernel, blur_kernel), 0)
        image_bgr[y:y+h, x:x+w] = blurred_patch

        blurred_rgb = cv2.cvtColor(blurred_patch, cv2.COLOR_BGR2RGB)
        blurred_image[y:y+h, x:x+w] = blurred_rgb
        blurred_faces.add(idx)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Function to reset image
def reset_image():
    global original_image, blurred_faces, blurred_image, detected_faces
    if original_image is None:
        return None
    blurred_faces.clear()
    blurred_image = original_image.copy()

    image_bgr = cv2.cvtColor(original_image.copy(), cv2.COLOR_RGB2BGR)
    for idx, face in enumerate(detected_faces):
        x, y, w, h = face['box']
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image_bgr, str(idx + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    reset_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return reset_rgb

# Function to save the blurred image to a temporary file and return it for download
def save_blurred_image():
    global blurred_image
    if blurred_image is None:
        return None
    # Save the blurred image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file.name, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
    return temp_file.name

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Face Blur App 🎭 (Click on face to Blur/Unblur Faces)")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an Image")
        image_output = gr.Image(type="numpy", label="Detected Faces", interactive=True)

    blur_slider = gr.Slider(minimum=1, maximum=1000, value=99, step=1, label="Blur Intensity")

    with gr.Row():
        detect_button = gr.Button("Detect Faces Before Blurring Them Yourself!")
        blur_all_button = gr.Button("Blur All Faces")

    with gr.Row():
        reset_button = gr.Button("Reset Image!")
        show_blur_button = gr.Button("Show Blurred Image Result!")

    blurred_image_output = gr.Image(type="numpy", label="Blurred Image")

    download_button = gr.Button("Download Blurred Image with original file format!")
    download_output = gr.File(label="Download Image")

    detect_button.click(detect_faces, inputs=image_input, outputs=image_output)
    reset_button.click(reset_image, outputs=image_output)

    image_output.select(toggle_blur_face, inputs=[image_output, blur_slider], outputs=image_output)
    blur_all_button.click(blur_all_faces, inputs=[image_output, blur_slider], outputs=image_output)
    show_blur_button.click(save_blurred_image, outputs=blurred_image_output)
    download_button.click(save_blurred_image, outputs=download_output)

# Launch App
demo.launch()
