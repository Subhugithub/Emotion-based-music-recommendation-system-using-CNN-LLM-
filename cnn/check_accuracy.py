from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "emotion_model.h5"
TEST_DIR = "dataset/fer2013/test"

# Load model
model = load_model(MODEL_PATH)

# Image preprocessing
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# Evaluate
loss, accuracy = model.evaluate(test_data)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
