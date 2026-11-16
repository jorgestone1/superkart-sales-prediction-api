import os

import pandas as pd
import requests
import streamlit as st

# The backend is reached over the Docker network by its container name; override with an env var if needed
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860")

# The four stores in the training data. Picking a store fills its attributes together,
# so the form cannot describe a store profile the model has never seen.
STORES = {
    "OUT001 - Supermarket Type1, Tier 2, High (est. 1987)":
        {"Store_Size": "High", "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type1", "Store_Age_Years": 38},
    "OUT002 - Food Mart, Tier 3, Small (est. 1998)":
        {"Store_Size": "Small", "Store_Location_City_Type": "Tier 3", "Store_Type": "Food Mart", "Store_Age_Years": 27},
    "OUT003 - Departmental Store, Tier 1, Medium (est. 1999)":
        {"Store_Size": "Medium", "Store_Location_City_Type": "Tier 1", "Store_Type": "Departmental Store", "Store_Age_Years": 26},
    "OUT004 - Supermarket Type2, Tier 2, Medium (est. 2009)":
        {"Store_Size": "Medium", "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2", "Store_Age_Years": 16},
}
PRODUCT_GROUPS = {"FD - Food": "FD", "DR - Drinks": "DR", "NC - Non-consumables": "NC"}

st.set_page_config(page_title="SuperKart Sales Prediction", layout="centered")
st.title("SuperKart Sales Prediction")
st.write("Predicts the total sales revenue of a product in a SuperKart store.")

# ---------------- Online prediction ----------------
st.subheader("Online Prediction")

store_label = st.selectbox("Store", list(STORES.keys()), index=3)
store = STORES[store_label]
st.caption(f"Size: {store['Store_Size']} | City tier: {store['Store_Location_City_Type']} | "
           f"Type: {store['Store_Type']} | Age: {store['Store_Age_Years']} years")

group_label = st.selectbox("Product group (first two letters of the Product ID)", list(PRODUCT_GROUPS.keys()))
product_id_char = PRODUCT_GROUPS[group_label]

# In the training data every non-consumable is "No Sugar", and only food items are perishable
if product_id_char == "NC":
    sugar = "No Sugar"
    category = "Non Perishables"
    st.caption("Non-consumables are recorded as 'No Sugar' and 'Non Perishables'.")
else:
    sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular"])
    category = (st.selectbox("Product category", ["Perishables", "Non Perishables"],
                             help="Perishables: Dairy, Meat, Fruits and Vegetables, Breads, Breakfast, Seafood")
                if product_id_char == "FD" else "Non Perishables")

weight = st.number_input("Product weight", min_value=1.0, max_value=30.0, value=12.66, step=0.1)
mrp = st.number_input("Product MRP (maximum retail price)", min_value=1.0, max_value=500.0, value=147.0, step=1.0)
area = st.number_input("Allocated display area (ratio of the store's total)", min_value=0.001, max_value=0.5,
                       value=0.056, step=0.001, format="%.3f")

record = {
    "Product_Weight": weight,
    "Product_Sugar_Content": sugar,
    "Product_Allocated_Area": area,
    "Product_MRP": mrp,
    "Store_Size": store["Store_Size"],
    "Store_Location_City_Type": store["Store_Location_City_Type"],
    "Store_Type": store["Store_Type"],
    "Product_Id_char": product_id_char,
    "Store_Age_Years": store["Store_Age_Years"],
    "Product_Type_Category": category,
}

if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=record, timeout=30)
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")
    else:
        result = response.json()
        if response.status_code == 200:
            st.success(f"Predicted sales revenue: {result['Predicted_Product_Store_Sales_Total']:,.2f}")
            for note in result.get("warnings", []):
                st.warning(note)
        else:
            st.error(f"API error ({response.status_code}): {result.get('error')}")

# ---------------- Batch prediction ----------------
st.subheader("Batch Prediction")
st.write("Upload a CSV with the ten feature columns used by the model.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded is not None and st.button("Predict for Batch", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                 files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")}, timeout=60)
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")
    else:
        result = response.json()
        if response.status_code == 200:
            inputs = pd.read_csv(uploaded)
            predictions = result["predictions"]
            inputs["Predicted_Sales"] = [predictions[str(i)] for i in inputs.index]
            st.success(f"Predictions completed for {result['rows']} rows.")
            for note in result.get("warnings", []):
                st.warning(note)
            st.dataframe(inputs)
            st.download_button("Download predictions as CSV", inputs.to_csv(index=False).encode("utf-8"),
                               file_name="superkart_batch_predictions.csv", mime="text/csv")
        else:
            st.error(f"API error ({response.status_code}): {result.get('error')}")
