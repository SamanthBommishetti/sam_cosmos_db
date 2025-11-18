from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from pymongo import MongoClient, ReplaceOne
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
    title="Product API + ML Price Prediction",
    description="Cosmos DB + FastAPI + Auto-fill missing prices with ML",
    version="1.0"
)


class ProductIn(BaseModel):
    name: str
    category: str
    price: Optional[float] = None
    quantity: Optional[int] = None
    inStock: bool = True
    description: Optional[str] = None

class ProductOut(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    category: str
    price: Optional[float] = None
    inStock: bool
    quantity: Optional[int] = None
    description: Optional[str] = None

    class Config:
        allow_population_by_field_name = True




@app.get("/")
def home():
    return {"message": "Product API is running! Go to /docs"}



@app.get("/products/", response_model=List[ProductOut])
def get_products(
    in_stock: Optional[bool] = None,
    limit: Optional[int] = Query(50, ge=1, le=200, description="Max 200 products")
):
    query = {}
    if in_stock is not None:
        query["inStock"] = in_stock

    cursor = collection.find(query).limit(limit or 50)
    products = list(cursor)

   
    for p in products:
        p["_id"] = str(p["_id"])
        for field in ["quantity", "description", "price"]:
            if field not in p or pd.isna(p[field]):
                p[field] = None

    return products



@app.post("/products/", response_model=ProductOut, status_code=201)
def create_product(product: ProductIn):
    doc = product.dict()
    doc["_id"] = os.urandom(12).hex() 
    collection.insert_one(doc)
    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: str):
    doc = collection.find_one({"_id": product_id})
    if not doc:
        raise HTTPException(404, "Product not found")
    doc["_id"] = str(doc["_id"])
    return doc



@app.put("/products/{product_id}", response_model=ProductOut)
def update_product(product_id: str, updates: ProductIn):
    update_data = {k: v for k, v in updates.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(400, "No data to update")

    result = collection.update_one({"_id": product_id}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(404, "Product not found")

    updated = collection.find_one({"_id": product_id})
    updated["_id"] = str(updated["_id"])
    return updated



@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: str):
    result = collection.delete_one({"_id": product_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Product not found")
    return None



@app.post("/ml/predict-missing-prices")
def predict_missing_prices():
    docs = list(collection.find({}))
    if not docs:
        return {"message": "No products found"}

    df = pd.DataFrame(docs)
    missing_count = df["price"].isna().sum()
    if missing_count == 0:
        return {"message": "No missing prices to predict"}

    # Feature Engineering
    le = LabelEncoder()
    df["cat_encoded"] = le.fit_transform(df["category"].astype(str))
    df["inStock_num"] = df["inStock"].astype(int)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["name_len"] = df["name"].astype(str).apply(len)

    features = ["cat_encoded", "inStock_num", "quantity", "name_len"]
    train = df[df["price"].notna()]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[features], train["price"])

    predict_df = df[df["price"].isna()]
    if len(predict_df) > 0:
        preds = np.round(model.predict(predict_df[features]), 2)
        for idx, row in predict_df.iterrows():
            pred_price = float(preds[predict_df.index.get_loc(idx)])
            collection.update_one(
                {"_id": row["_id"]},
                {"$set": {"price": pred_price}}
            )

    return {"message": f"Success! Filled {len(predict_df)} missing prices with ML"}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
