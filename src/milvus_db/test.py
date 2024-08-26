from pymilvus import Milvus, IndexType, connections, Collection
import numpy as np
import os

def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def find_nearest_outfits(item1_vector, item2_vector, top_k=1):
    search_params = {"metric_type": MetricType.L2, "params": {"nprobe": 10}}
    
    # Поиск ближайшего наряда для первого элемента
    _, results = client.search(collection_name='outfits', query_records=[item1_vector], top_k=top_k, params=search_params)
    nearest_outfit1 = results[0].ids[0]
    
    # Поиск ближайшего наряда для второго элемента
    _, results = client.search(collection_name='outfits', query_records=[item2_vector], top_k=top_k, params=search_params)
    nearest_outfit2 = results[0].ids[0]
    
    return nearest_outfit1, nearest_outfit2

# Расчет меры совместимости
def calculate_compatibility_diversity(item1_vector, item2_vector):
    nearest_outfit1, nearest_outfit2 = find_nearest_outfits(item1_vector, item2_vector)
    
    # Вычисление сходства между найденными нарядами
    outfit1_vector = client.get_entity_by_id('outfits', [nearest_outfit1])[0].vectors[0]
    outfit2_vector = client.get_entity_by_id('outfits', [nearest_outfit2])[0].vectors[0]
    outfit_similarity = np.dot(outfit1_vector, outfit2_vector) / (np.linalg.norm(outfit1_vector) * np.linalg.norm(outfit2_vector))
    
    # Вычисление разнообразия между элементами одежды
    item_diversity = 1 - np.dot(item1_vector, item2_vector) / (np.linalg.norm(item1_vector) * np.linalg.norm(item2_vector))
    
    # Расчет меры совместимости
    compatibility_diversity = outfit_similarity * item_diversity
    
    return compatibility_diversity