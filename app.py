import streamlit as st
import face_recognition
import cv2
import numpy as np

def calculate_similarity(image1, image2):
    """
    Calculates the similarity between two images using face_recognition and returns whether the faces are similar.
    """
    # Convert images to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations1 = face_recognition.face_locations(image1_rgb)
    face_locations2 = face_recognition.face_locations(image2_rgb)

    if not face_locations1 or not face_locations2:
        return False, 0, False

    # Extract face encodings
    face_encodings1 = face_recognition.face_encodings(image1_rgb, face_locations1)[0]
    face_encodings2 = face_recognition.face_encodings(image2_rgb, face_locations2)[0]

    # Compare faces
    results = face_recognition.compare_faces([face_encodings1], face_encodings2)
    face_distance = face_recognition.face_distance([face_encodings1], face_encodings2)[0]

    similarity_threshold = 0.6
    is_similar = face_distance < similarity_threshold

    return True, face_distance, is_similar

def main():
    st.title("Face Similarity Checker")

    file1 = st.file_uploader("Choose the first image", type=["jpg", "jpeg", "png"])
    file2 = st.file_uploader("Choose the second image", type=["jpg", "jpeg", "png"])

    if file1 and file2:
        # Convert the uploaded files to OpenCV images
        file_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
        image1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)

        file_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
        image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

        # Show uploaded images
        st.image(image1, channels="BGR", caption="First Image")
        st.image(image2, channels="BGR", caption="Second Image")

        # Calculate similarity
        has_faces, similarity_score, is_similar = calculate_similarity(image1, image2)

        if not has_faces:
            st.error("No faces detected in one or both images.")
        else:
            st.success(f"Faces Detected! Similarity Score: {similarity_score:.2f}")
            if is_similar:
                st.success("The faces are likely similar.")
            else:
                st.error("The faces are likely not similar.")

if __name__ == "__main__":
    main()
