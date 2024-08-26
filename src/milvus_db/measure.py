from pymilvus import Milvus, IndexType, Collection
import numpy as np
import os
from milvus_functions import connect_to_milvus, create_collection, create_index, find_nearest_outfits

connect_to_milvus()

data_dir = '/home/doodhust/sourse/new_summer_project_DNS/datavectors/clothes' # 2 файла npy на вход

vector1 = None
vector2 = None

file_count = 0
for filename in os.listdir(data_dir):
    if filename.endswith('.npy'):
        file_path = os.path.join(data_dir, filename)
        vector_data = np.load(file_path)
        if file_count == 0:
            vector1 = vector_data
        elif file_count == 1:
            vector2 = vector_data
        file_count += 1
        if file_count == 2:
            break


print(find_nearest_outfits(vector1, vector2))




