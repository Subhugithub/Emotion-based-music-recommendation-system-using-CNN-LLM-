# 🎵 Emotion-Based Music Recommendation System

## 📌 Project Overview
This project is an Emotion-Based Music Recommendation System that detects user emotions using facial expressions and text input, and recommends music accordingly in real time.

The system uses:
- CNN (Convolutional Neural Network) for facial emotion recognition  
- LLM (Large Language Model) for text emotion analysis  

---

## 🚀 Features
- Real-time emotion detection using webcam  
- Facial emotion recognition using CNN  
- Text-based emotion detection using LLM  
- Hybrid approach (CNN + LLM)  
- Automatic music recommendation using YouTube  
- User-friendly interface using Streamlit  

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- Streamlit  
- NumPy  

---

## 📂 Project Structure
emotion-music-recommendation-system/
│
├── app.py
├── cnn/
├── llm/
├── requirements.txt
└── README.md

---

## ⚙️ How to Run

1. Clone the repository:
git clone https://github.com/Subhugithub/Emotion-based-music-recommendation-system-using-CNN-LLM-.git

2. Go to project folder:
cd emotion-music-recommendation-system

3. Install dependencies:
pip install -r requirements.txt

4. Run the app:
streamlit run app.py

---

## 📊 Dataset
- FER-2013 dataset  
- 48x48 grayscale images  
- 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  

---

## 🧠 Working (Methodology)
1. Capture image using webcam  
2. Detect face using OpenCV  
3. Predict emotion using CNN model  
4. Analyze text emotion using LLM  
5. Combine both results  
6. Recommend music using YouTube  

---

## 📈 Performance
- Improved accuracy using hybrid approach  
- Real-time emotion detection  
- Better user experience  

---

## ⚠️ Limitations
- Dataset imbalance  
- Lighting affects accuracy  
- Complex emotions are difficult to detect  

---

## 🔮 Future Work
- Voice emotion detection  
- Multi-language support  
- Spotify integration  
- Better deep learning models  

---

## 👨‍💻 Author
Shubham Kumar  
MCA Final Year Project

---

## 📎 Note
Model file (.h5) is not uploaded due to large size.
