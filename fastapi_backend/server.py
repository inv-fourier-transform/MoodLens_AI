from fastapi import FastAPI, File, UploadFile
from model_helper import detect_emotion
import os

app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions from images with soft label probabilities",
    version="1.1.0"
)


@app.get("/hello")
async def hello():
    return {"message": "Hello Emotion World"}


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Upload an image and get emotion prediction with probability distribution.
    """
    temp_path = "temp_file.jpg"

    try:
        # Read uploaded file
        image_bytes = await file.read()

        # Save temporarily
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        # Get prediction (returns dictionary with hard + soft labels)
        result = detect_emotion(temp_path)

        # Structure response
        return {
            "status": "success",
            "prediction": {
                "dominant_emotion": result['hard_label'],
                "confidence": round(result['confidence'], 4),
                "probability_distribution": result['soft_probabilities']
            },
            "all_emotions": result['all_emotions']
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)



