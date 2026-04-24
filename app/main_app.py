import cv2
import numpy as np
import webbrowser
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("../cnn/emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion to YouTube mapping
emotion_song_map = {
    "Happy": "happy songs playlist",
    "Sad": "sad songs playlist",
    "Angry": "angry rock songs",
    "Surprise": "party songs",
    "Neutral": "chill music",
    "Fear": "relaxing music",
    "Disgust": "calm instrumental music"
}

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
detected_emotion = None

print("Press 's' to recommend song, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        prediction = model.predict(roi, verbose=0)
        confidence = np.max(prediction)

        if confidence > 0.5:
            detected_emotion = emotion_labels[np.argmax(prediction)]
        else:
            detected_emotion = "Neutral"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, detected_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Emotion Based Music Recommendation", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press S to open YouTube
    if key == ord('s') and detected_emotion:
        query = emotion_song_map.get(detected_emotion, "relaxing music")
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(url)
        print(f"YouTube opened for emotion: {detected_emotion}")

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
