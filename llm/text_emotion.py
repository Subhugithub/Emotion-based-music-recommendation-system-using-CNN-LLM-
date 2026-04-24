def detect_emotion_from_text(text):
    text = text.lower()

    if any(word in text for word in ["happy", "joy", "excited", "great"]):
        return "Happy"
    elif any(word in text for word in ["sad", "depressed", "cry", "lonely"]):
        return "Sad"
    elif any(word in text for word in ["angry", "mad", "furious"]):
        return "Angry"
    elif any(word in text for word in ["surprise", "shocked", "wow"]):
        return "Surprise"
    elif any(word in text for word in ["fear", "scared", "afraid"]):
        return "Fear"
    else:
        return "Neutral"
