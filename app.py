import os
import cv2
import numpy as np
import streamlit as st
from tensorflow import keras
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input

# Placeholder function to preprocess the input image similar to training images
model = keras.models.load_model('Attendance_Gender.h5')


def preprocess_input_image(file_stream):
    # Convert the file stream to a numpy array
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)

    # Read the image from the numpy array
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize the image
    img = tf.image.resize(img, (256, 256))

    img = np.expand_dims(img/255, 0)
    return img


# Assuming the model predicts binary classes (0 for Female, 1 for Male)
def predict_gender(features_flat):
    # features_flat = preprocess_input_features(features_flat)
    prediction = model.predict(features_flat)
    gender = "Male" if prediction[0][0] > 0.65 else "Female"
    # confidence = round(prediction[0][0], 2) if prediction[0][0] > 0.5 else round(
    #     (1 - prediction[0][0]), 2)

    from decimal import Decimal, ROUND_HALF_UP

    confidence = Decimal(str(prediction[0][0]))

    # If the confidence is greater than 0.5, round normally; otherwise, round the complement
    confidence = confidence.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP) if confidence > 0.65 else confidence.quantize(
        Decimal('0.00'), rounding=ROUND_HALF_UP)

    return (gender, confidence)

# Placeholder function to dynamically choose the database based on gender


def choose_database(predicted_gender):
    if predicted_gender == "Male":
        return 'Male'
    else:
        return 'Female'


# Initialize session state
session_state = st.session_state
if 'session_state' not in session_state:
    session_state.session_state = False
    session_state.attendance_list = []
    # Add a new variable to control the recording loop
    session_state.end_recording = False

# Set Streamlit app title and favicon
st.set_page_config(
    page_title="Attendance Marking System",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="auto"
)

# App title
st.title('Intro to AI Final Project: Fingerprint Attendance Marking System')

# Description
st.write(
    "This app uses advanced algorithms for gender prediction and fingerprint matching, for quick biometric identification."
)

# Authors' names and course description
st.markdown("Authors: Ryan Mbun Tangwe and Sombang Patience")

# Begin attendance recording button
if not session_state.session_state:
    begin_button_clicked = st.button("Begin Record Attendance")

    # Check if the "Begin Record Attendance" button is clicked
    if begin_button_clicked:
        # Set session state to True
        session_state.session_state = True
else:
    # Display an empty text in place of the button
    st.text("")

# Check if the attendance recording session is ongoing
if session_state.session_state:
    # Always display the "End Recording" button at the top
    if st.button("End Recording", key="End"):
        session_state.end_recording = True

    uploaded_file = st.file_uploader("To Mark your attendance, Upload your fingerprint image: ", type=[
        "bmp", "jpg", "jpeg", "png"])

    if uploaded_file:
        fingerprint_file_name = uploaded_file.name
        fingerprint_path = ("All/" + fingerprint_file_name)
        print("fingerprint path:", fingerprint_path)

        features_flat = preprocess_input_image(uploaded_file)

        predicted_gender = predict_gender(features_flat)[0]
        confidence = predict_gender(features_flat)[1]
        st.write(f"Predicted Gender: {predicted_gender}")
        st.write(f"Confidence: {confidence}")

        chosen_database = choose_database(predicted_gender)

        class_name = "All"
        print("Read as sample: " + class_name + "/" + fingerprint_file_name)
        sample = cv2.imread(class_name + "/" + fingerprint_file_name)
        best_score = 0
        filename = None
        image = None
        kp1, kp2, mp = None, None, None

        # Simulate fingerprint collection session incrementally
        fingerprint_collection_complete = False

        # Fingerprint matching logic
        for file in [file for file in os.listdir(chosen_database) if not file.startswith('desktop')]:
            fingerprint_image = cv2.imread(chosen_database + "/" + file)
            sift = cv2.SIFT_create()

            keypoints_1, descriptors_1 = sift.detectAndCompute(
                sample, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(
                fingerprint_image, None)

            matches = cv2.BFMatcher().knnMatch(descriptors_1, descriptors_2, k=2)
            match_points = []

            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    match_points.append(p)

            keypoint = min(len(keypoints_1), len(keypoints_2))

            if keypoint > 0:
                current_score = len(match_points) / keypoint * 100

                if current_score > best_score:
                    best_score = current_score
                    filename = file
                    image = fingerprint_image
                    kp1, kp2, mp = keypoints_1, keypoints_2, match_points

                    if filename is not None:
                        # Display unknown fingerprint
                        col1, col2 = st.columns(2)
                        col1.image(
                            sample, caption="Unknown Fingerprint", width=250)

                        # Display Matched fingerprint
                        col2.image(
                            image, caption=f"Best Match: {filename[:-4]}", width=250)

                        # Display the matching points
                        st.image(
                            cv2.drawMatches(
                                sample, kp1, fingerprint_image, kp2, mp, None),
                            caption=f"Matching Points with {filename[:-4]}", width=500)

                        # Record attendance
                        session_state.attendance_list.append(filename[:-4])

                    else:
                        st.write("No match found.")

            if session_state.end_recording:
                break
    else:
        st.write("Please upload a fingerprint image.")

    if session_state.end_recording:
        # End the attendance recording session
        session_state.session_state = False
        st.write("Attendance Recording Ended.")
        st.write("Saving attendance to CSV...")

        # Collect filenames of the files in the class_name folder
        attendance_folder = os.listdir(class_name)

        # Save attendance to CSV
        attendance_data = []
        for filename in [filename for filename in attendance_folder if not filename.startswith('desktop')]:
            student_id, student_name = filename.split(
                "-")[0], filename.split("-")[1][:-4]
            attendance_status = "Present" if filename[:-
                                                      4] in session_state.attendance_list else "Absent"
            attendance_data.append(
                (student_name, student_id, attendance_status))

        # Create a DataFrame and save to CSV
        import pandas as pd
        df = pd.DataFrame(attendance_data, columns=[
                          'Name', 'ID', 'Attendance'])
        df.to_csv('attendance.csv', index=False)

        st.write("Attendance saved to 'attendance.csv'.")
else:
    st.write(
        "Click 'Begin Record Attendance' to start the attendance recording session.")

# Display the footer or any additional information
st.write("© 2023 PENIEL .inc\nAll rights reserved.")
