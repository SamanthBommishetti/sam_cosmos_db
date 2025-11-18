from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
import time
load_dotenv()
 
CONN_STRING = os.getenv("COSMOS_CONN_STRING")
DB_NAME = os.getenv("DB_NAME")
COLL_NAME = os.getenv("COLLECTION")
 
client = MongoClient(CONN_STRING, tls=True, tlsAllowInvalidCertificates=True)
 
db = client[DB_NAME]
collection = db[COLL_NAME]
 
# Insert
def create_product():
    cars=pd.read_csv("products_100_with_quantity.csv")
    cars_dicts=cars.to_dict(orient="records")
    for i in range(0, len(cars_dicts), 30):
        batch = cars_dicts[i:i + 30]
        result=collection.insert_many(batch)
        time.sleep(0.2)
    print("Created an Item ID:", result.inserted_ids)

# Read
def read_product():
    cursor = collection.find({"category": "Electronics"})
    print("Electronic Items:")
    for doc in cursor:
        print(doc)

if __name__ == "__main__":
    print("Connected to Cosmos DB Mongo API")
    create_product()
    # update_item()
    # delete_product()
    read_product()
