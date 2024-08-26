from pymilvus import connections, IndexType, FieldSchema, CollectionSchema, Collection, DataType, utility
import numpy as np
import os

def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_collection(name, fields):
    if utility.has_collection(name):
        print(f"Collection '{name}' already exists. Deleting it.")
        utility.drop_collection(name)
        print(f"Collection '{name}' deleted.")

    schema = CollectionSchema(fields)
    collection = Collection(name, schema, consistency_level="Strong")
    print(f"Collection '{name}' created.")
    return collection

def create_index(collection):
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {
            "nlist": 1024
        }
    }

    collection.create_index(
    field_name="vector",
    index_params=index_params,
    index_name="scalar_index"
    )
    print(f'Index created')

def find_nearest_outfits(vector1, vector2):
    collection = Collection("clothing_collection")
    collection.load()

    search_params = {
        "metric_type": "L2",
        "offset": 5,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }

    results1 = collection.search(
        data=[vector1], 
        anns_field="vector", 
        param=search_params,
        limit=3,
        expr=None,
        consistency_level="Strong"
    )

    results2 = collection.search(
        data=[vector2], 
        anns_field="vector", 
        param=search_params,
        limit=3,
        expr=None,
        consistency_level="Strong"
    )

    return results1, results2