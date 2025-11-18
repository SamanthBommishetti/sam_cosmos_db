from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time
import uvicorn

load_dotenv()

CONN_STRING = os.getenv("COSMOS_CONN_STRING")
DB_NAME = os.getenv("DB_NAME", "ecommerceDB")
COLL_NAME = os.getenv("COLLECTION", "products")

client = MongoClient(CONN_STRING, tls=True, tlsAllowInvalidCertificates=True)
db = client[DB_NAME]
collection = db[COLL_NAME]


app = FastAPI(
    title="Product API with ML Price Prediction",
    description="CRUD + Auto-fill missing prices using Random Forest",
    version="1.0"
)

class ProductIn(BaseModel):
    name: str
    category: str
    price: Optional[float] = None
    quantity: int = Field(..., ge=0)
    inStock: bool = True
    description: Optional[str] = ""

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    inStock: Optional[bool] = None
    description: Optional[str] = None

class ProductOut(ProductIn):
    id: str = Field(..., alias="_id")

def predict_and_fill_missing_prices():
    all_products = list(collection.find({}))
    df = pd.DataFrame(all_products)

    if df.empty or df["price"].isna().sum() == 0:
        return {"message": "No missing prices to fill"}

  
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"].astype(str))
    df["inStock_num"] = df["inStock"].astype(int)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["name_length"] = df["name"].astype(str).apply(len)

    features = ["category_encoded", "inStock_num", "quantity", "name_length"]

    train_df = df[df["price"].notna()]
    X_train = train_df[features]
    y_train = train_df["price"]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predict_df = df[df["price"].isna()]
    X_predict = predict_df[features]
    predicted = np.round(model.predict(X_predict), 2)

    updated = 0
    for idx, row in predict_df.iterrows():
        pred_price = float(predicted[predict_df.index.get_loc(idx)])
        result = collection.update_one(
            {"_id": row["_id"]},
            {"$set": {"price": pred_price}}
        )
        if result.modified_count:
            updated += 1

    return {"message": f"ML Prediction Complete! Filled {updated} missing prices"}


@app.post("/products/", response_model=ProductOut, status_code=201)
def create_product(product: ProductIn):
    doc = product.dict()
    doc["_id"] = os.urandom(12).hex() 
    result = collection.insert_one(doc)
    if result.inserted_id:
        doc["_id"] = str(doc["_id"])
        return doc
    raise HTTPException(500, "Failed to insert")


@app.get("/products/", response_model=List[ProductOut])
def get_products(category: Optional[str] = None, in_stock: Optional[bool] = None, limit: int = 50):
    query = {}
    if category:
        query["category"] = category
    if in_stock is not None:
        query["inStock"] = in_stock

    cursor = collection.find(query).limit(limit)
    products = list(cursor)
    for p in products:
        p["_id"] = str(p["_id"])
    return products


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: str):
    doc = collection.find_one({"_id": product_id})
    if not doc:
        raise HTTPException(404, "Product not found")
    doc["_id"] = str(doc["_id"])
    return doc

@app.put("/products/{product_id}", response_model=ProductOut)
def update_product(product_id: str, updates: ProductUpdate):
    update_data = {k: v for k, v in updates.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(400, "No data provided to update")

    result = collection.update_one({"_id": product_id}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(404, "Product not found or no changes made")

    updated_doc = collection.find_one({"_id": product_id})
    updated_doc["_id"] = str(updated_doc["_id"])
    return updated_doc


@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: str):
    result = collection.delete_one({"_id": product_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Product not found")
    return None


@app.post("/ml/predict-missing-prices")
def ml_predict_prices():
    result = predict_and_fill_missing_prices()
    return result


@app.get("/")
def home():
    return {"message": "Product API with ML Price Prediction is running!", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)