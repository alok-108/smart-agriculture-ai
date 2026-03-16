"""
FastAPI backend for Smart Agriculture AI System
Provides endpoints for crop recommendation, yield prediction,
fertilizer recommendation, and plant disease classification.
"""
import os
import logging
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Agriculture AI API",
    description="REST API for crop recommendation, yield prediction, fertilizer recommendation, and disease detection",
    version="1.0.0"
)

# ------------------------------------------------------------------
# Load models and encoders at startup
# ------------------------------------------------------------------
MODELS_DIR = "saved_models"

# Crop recommendation
try:
    rf_crop = joblib.load(os.path.join(MODELS_DIR, "rf_crop_recommendation.pkl"))
    le_crop = joblib.load(os.path.join(MODELS_DIR, "crop_label_encoder.pkl"))
    logger.info("Crop recommendation model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load crop recommendation model: {e}")
    rf_crop = le_crop = None

# Yield prediction
try:
    rf_yield = joblib.load(os.path.join(MODELS_DIR, "rf_yield_prediction.pkl"))
    logger.info("Yield prediction model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load yield prediction model: {e}")
    rf_yield = None

# Fertilizer recommendation (SVM + encoders)
try:
    svm_fert = joblib.load(os.path.join(MODELS_DIR, "svm_fertility_classifier.pkl"))
    le_soil = joblib.load(os.path.join(MODELS_DIR, "soil_type_encoder.pkl"))
    le_crop_type = joblib.load(os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
    le_fert = joblib.load(os.path.join(MODELS_DIR, "fert_label_encoder.pkl"))
    logger.info("Fertilizer recommendation model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load fertilizer recommendation model: {e}")
    svm_fert = le_soil = le_crop_type = le_fert = None

# Disease detection
try:
    disease_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "crop_disease_model.h5"))
    # IMPORTANT: Provide the list of class names corresponding to the disease model.
    # These should be the 131 fruit classes used during training.
    # Replace with your actual class names if different.
    disease_class_names = [
        "Apple Braeburn", "Apple Crimson Snow", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3",
        "Apple Granny Smith", "Apple Pink Lady", "Apple Red 1", "Apple Red 2", "Apple Red 3",
        "Apple Red Delicious", "Apple Red Yellow 1", "Apple Red Yellow 2", "Apricot", "Avocado",
        "Avocado ripe", "Banana", "Banana Red", "Cactus fruit", "Cantaloupe 1", "Cantaloupe 2",
        "Carambula", "Cherry 1", "Cherry 2", "Cherry Rainier", "Cherry Wax Black", "Cherry Wax Red",
        "Cherry Wax Yellow", "Chestnut", "Clementine", "Cocos", "Corn", "Corn Husk", "Cucumber Ripe",
        "Cucumber Ripe 2", "Dates", "Eggplant", "Fig", "Ginger Root", "Granadilla", "Grape Blue",
        "Grape Pink", "Grape White", "Grape White 2", "Grapefruit Pink", "Grapefruit White",
        "Guava", "Hazelnut", "Huckleberry", "Kaki", "Kiwi", "Kohlrabi", "Kumquats", "Lemon",
        "Lemon Meyer", "Limes", "Lychee", "Mandarine", "Mango", "Maracuja", "Melon Piel de Sapo",
        "Mulberry", "Nectarine", "Nectarine Flat", "Nut Forest", "Nut Pecan", "Onion Red",
        "Onion Red Peeled", "Onion White", "Orange", "Papaya", "Passion Fruit", "Peach", "Peach 2",
        "Peach Flat", "Pear", "Pear 2", "Pear Abate", "Pear Forelle", "Pear Kaiser", "Pear Monster",
        "Pear Red", "Pear Stone", "Pear Williams", "Pepino", "Pepper Green", "Pepper Orange",
        "Pepper Red", "Pepper Yellow", "Physalis", "Physalis with Husk", "Pineapple", "Pineapple Mini",
        "Pitahaya Red", "Plum", "Plum 2", "Plum 3", "Pomegranate", "Pomelo Sweetie", "Potato Red",
        "Potato Red Washed", "Potato Sweet", "Potato White", "Quince", "Rambutan", "Raspberry",
        "Redcurrant", "Salak", "Strawberry", "Strawberry Wedge", "Tamarillo", "Tangelo", "Tomato 1",
        "Tomato 2", "Tomato 3", "Tomato 4", "Tomato Cherry Red", "Tomato Heart", "Tomato Maroon",
        "Tomato Yellow", "Walnut"
    ]
    logger.info("Disease detection model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load disease detection model: {e}")
    disease_model = None
    disease_class_names = []

# ------------------------------------------------------------------
# Pydantic models for request validation
# ------------------------------------------------------------------
class CropFeatures(BaseModel):
    N: float = Field(..., ge=0, description="Nitrogen content")
    P: float = Field(..., ge=0, description="Phosphorus content")
    K: float = Field(..., ge=0, description="Potassium content")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    ph: float = Field(..., ge=0, le=14, description="pH value")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")

    @validator('ph')
    def validate_ph(cls, v):
        if v < 0 or v > 14:
            raise ValueError('pH must be between 0 and 14')
        return v

class YieldFeatures(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float = Field(...)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)
    area: float = Field(..., ge=0, description="Area in hectares")

class FertilizerFeatures(BaseModel):
    Temparature: float = Field(..., alias="temperature", description="Temperature in Celsius")
    Humidity: float = Field(..., ge=0, le=100)
    Moisture: float = Field(..., ge=0, description="Soil moisture")
    Soil_Type: str = Field(..., alias="soil_type", description="Type of soil (e.g., Sandy, Loamy)")
    Crop_Type: str = Field(..., alias="crop_type", description="Type of crop (e.g., Maize, Wheat)")
    Nitrogen: float = Field(..., ge=0)
    Potassium: float = Field(..., ge=0)
    Phosphorous: float = Field(..., ge=0)

    class Config:
        allow_population_by_field_name = True

# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Smart Agriculture AI API is running"}

@app.post("/predict_crop")
async def predict_crop(features: CropFeatures):
    """
    Predict the best crop to plant based on soil and climate features.
    """
    if rf_crop is None or le_crop is None:
        raise HTTPException(status_code=503, detail="Crop recommendation model not available")

    try:
        # Prepare input array
        input_data = np.array([[features.N, features.P, features.K,
                                features.temperature, features.humidity,
                                features.ph, features.rainfall]])
        # Predict encoded label
        pred_encoded = rf_crop.predict(input_data)[0]
        # Decode to crop name
        crop_name = le_crop.inverse_transform([pred_encoded])[0]
        return {"recommended_crop": crop_name}
    except Exception as e:
        logger.error(f"Error in /predict_crop: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict_yield")
async def predict_yield(features: YieldFeatures):
    """
    Predict expected crop yield based on soil, climate, and area.
    """
    if rf_yield is None:
        raise HTTPException(status_code=503, detail="Yield prediction model not available")

    try:
        input_data = np.array([[features.N, features.P, features.K,
                                features.temperature, features.humidity,
                                features.rainfall, features.ph, features.area]])
        yield_pred = rf_yield.predict(input_data)[0]
        return {"predicted_yield": float(yield_pred)}
    except Exception as e:
        logger.error(f"Error in /predict_yield: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict_fertilizer")
async def predict_fertilizer(features: FertilizerFeatures):
    """
    Recommend the appropriate fertilizer based on soil parameters and crop type.
    """
    if None in (svm_fert, le_soil, le_crop_type, le_fert):
        raise HTTPException(status_code=503, detail="Fertilizer recommendation model not available")

    try:
        # Encode categorical features
        soil_encoded = le_soil.transform([features.Soil_Type])[0]
        crop_encoded = le_crop_type.transform([features.Crop_Type])[0]

        # Build feature vector in the exact order used during training:
        # Temparature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
        input_data = np.array([[features.Temparature, features.Humidity, features.Moisture,
                                soil_encoded, crop_encoded,
                                features.Nitrogen, features.Potassium, features.Phosphorous]])
        pred_encoded = svm_fert.predict(input_data)[0]
        fertilizer_name = le_fert.inverse_transform([pred_encoded])[0]
        return {"recommended_fertilizer": fertilizer_name}
    except ValueError as ve:
        # Handle unknown soil or crop types
        raise HTTPException(status_code=400, detail=f"Invalid soil or crop type: {ve}")
    except Exception as e:
        logger.error(f"Error in /predict_fertilizer: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload an image of a plant leaf to get the predicted disease/health class.
    """
    if disease_model is None:
        raise HTTPException(status_code=503, detail="Disease detection model not available")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and preprocess the image
        contents = await file.read()
        # Decode image using TensorFlow
        img = tf.image.decode_image(contents, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)  # Add batch dimension

        # Predict
        predictions = disease_model.predict(img)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])

        # Map to class name
        if predicted_idx < len(disease_class_names):
            class_name = disease_class_names[predicted_idx]
        else:
            class_name = f"Class_{predicted_idx}"

        return {
            "predicted_class": class_name,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error in /predict_disease: {e}")
        raise HTTPException(status_code=500, detail="Image processing or prediction failed")

# ------------------------------------------------------------------
# Run the app (for local testing)
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
