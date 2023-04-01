import cv2
import face_recognition

# Load the known images and encode their faces
known_image = face_recognition.load_image_file("/Users/farbodforotani/Desktop/known_face.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Open the laptop camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find the faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "Unknown"

        # If a match was found in the known images, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = "Known Person"

        face_names.append(name)

    # Draw boxes around the faces and label them
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognizer', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
