from pymilvus import connections, IndexType, FieldSchema, CollectionSchema, Collection, DataType, utility
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

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
        limit=1,
        output_fields=['vector'],
        expr=None,
        consistency_level="Strong"
    )

    results2 = collection.search(
        data=[vector2], 
        anns_field="vector", 
        param=search_params,
        limit=1,
        output_fields=['vector'],
        expr=None,
        consistency_level="Strong"
    )

    hit1, hit2 = results1[0][0], results2[0][0]

    return hit1.entity.get('vector'), hit2.entity.get('vector')

def compdiv(vector1, vector2):

    """
    Calculates the Comp-Diversity measure between two clothing item vectors.
    
    Parameters:
    vector1 (numpy.ndarray): Vector representation of the first clothing item.
    vector2 (numpy.ndarray): Vector representation of the second clothing item.
    
    Returns:
    float: The Comp-Diversity measure between the two clothing items.
    """
    
    nearest_outfit1, nearest_outfit2 = find_nearest_outfits(vector1, vector2)
    
    compatibility = cosine_similarity([vector1], [vector2])[0][0]
    
    # Calculate the diversity
    diversity = 1 - cosine_similarity([nearest_outfit1], [nearest_outfit2])[0][0]
    
    # Calculate the Comp-Diversity measure
    comp_div = compatibility * diversity
    
    print(f'compdiv = {round(comp_div, 4)}')