from pymilvus import connections, IndexType, FieldSchema, CollectionSchema, Collection, DataType, utility
import numpy as np
import os
from milvus_functions import connect_to_milvus, create_collection, create_index, compatibility_diversity


data_dir = '/home/doodhust/sourse/new_summer_project_DNS/datavectors/vectors'
collection_name = 'clothing_collection'

data = []
ids = []
vectors = []

for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        vectors.append(np.load(os.path.join(data_dir, filename)).tolist())
        ids.append(len(vectors))
dimension = len(vectors[0])

data.append(ids)
data.append(vectors)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
]

connect_to_milvus()

collection = create_collection(collection_name, fields)

create_index(collection)

mr = collection.insert(data)

print(f"Collection '{collection_name}' created and populated with vectors from NPY files.")



