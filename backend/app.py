import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

# Initialize the Flask app
superkart_api = Flask("SuperKart Sales Predictor")
superkart_api.json.sort_keys = False  # keep batch row order in the JSON response

# Load the serialized pipeline (preprocessing + tuned model) once, at startup
model = joblib.load("superkart_model.joblib")

# The input contract: the ten features the pipeline was trained on
NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
CATEGORICAL_FEATURES = ["Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
                        "Store_Type", "Product_Id_char", "Product_Type_Category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Categories seen in training, read from the fitted encoder so they cannot drift from the model
_encoder = model.named_steps["columntransformer"].named_transformers_["onehotencoder"]
KNOWN_CATEGORIES = {col: set(cats) for col, cats in zip(CATEGORICAL_FEATURES, _encoder.categories_)}

# The four stores in the training data: (size, city tier, type, age in years)
KNOWN_STORES = {
    ("High", "Tier 2", "Supermarket Type1", 38): "OUT001",
    ("Small", "Tier 3", "Food Mart", 27): "OUT002",
    ("Medium", "Tier 1", "Departmental Store", 26): "OUT003",
    ("Medium", "Tier 2", "Supermarket Type2", 16): "OUT004",
}


class InputError(ValueError):
    """Raised for a request the API can describe back to the caller."""


def validate(frame):
    """Check columns and types; return the cleaned frame and a list of warnings."""
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise InputError(f"Missing required field(s): {', '.join(missing)}")
    frame = frame[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        converted = pd.to_numeric(frame[col], errors="coerce")
        bad_rows = frame.index[converted.isna()].tolist()
        if bad_rows:
            raise InputError(f"Field '{col}' must be numeric (problem in row(s) {bad_rows[:10]})")
        frame[col] = converted
    if (frame["Product_MRP"] <= 0).any() or (frame["Product_Weight"] <= 0).any():
        raise InputError("Product_MRP and Product_Weight must be greater than zero")
    if ((frame["Product_Allocated_Area"] < 0) | (frame["Product_Allocated_Area"] > 1)).any():
        raise InputError("Product_Allocated_Area is a ratio and must be between 0 and 1")
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype(str).str.strip()

    warnings = []
    for col in CATEGORICAL_FEATURES:
        unseen = sorted(set(frame[col]) - KNOWN_CATEGORIES[col])
        if unseen:
            warnings.append(f"{col} value(s) {unseen} were not seen in training; "
                            "the model ignores them, so treat these predictions as low confidence")
    store_keys = list(zip(frame["Store_Size"], frame["Store_Location_City_Type"],
                          frame["Store_Type"], frame["Store_Age_Years"].round().astype(int)))
    unknown_store_rows = [i for i, key in zip(frame.index, store_keys) if key not in KNOWN_STORES]
    if unknown_store_rows:
        warnings.append(f"{len(unknown_store_rows)} row(s) describe a store profile that matches none of the "
                        "four stores in the training data; those predictions are extrapolations")
    return frame, warnings


@superkart_api.errorhandler(InputError)
def handle_input_error(err):
    return jsonify({"error": str(err)}), 400


@superkart_api.errorhandler(Exception)
def handle_unexpected_error(err):
    if isinstance(err, HTTPException):  # keep 404 / 405 as they are
        return jsonify({"error": err.name, "detail": err.description}), err.code
    return jsonify({"error": "Prediction failed", "detail": str(err)}), 500


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok", "model": type(model[-1]).__name__, "features": FEATURES})


@superkart_api.post("/v1/predict")
def predict_sales():
    """Online inference: one product-store record as JSON in, one prediction out."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise InputError("Request body must be a JSON object with the ten feature fields")
    frame, warnings = validate(pd.DataFrame([payload]))
    prediction = round(float(model.predict(frame)[0]), 2)
    return jsonify({"Predicted_Product_Store_Sales_Total": prediction, "warnings": warnings})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    """Batch inference: a CSV file in, one prediction per row out, keyed by row index."""
    if "file" not in request.files:
        raise InputError("Upload the CSV under the form field 'file'")
    try:
        frame = pd.read_csv(request.files["file"])
    except Exception as exc:
        raise InputError(f"Could not read the uploaded file as CSV: {exc}")
    if frame.empty:
        raise InputError("The uploaded CSV has no rows")
    frame, warnings = validate(frame)
    predictions = np.round(model.predict(frame).astype(float), 2).tolist()
    return jsonify({"predictions": dict(zip([str(i) for i in frame.index], predictions)),
                    "rows": len(predictions), "warnings": warnings})


# Run the Flask development server when executed directly (the container uses gunicorn instead)
if __name__ == "__main__":
    superkart_api.run(debug=True)
