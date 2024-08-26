import numpy as np
import os
from PIL import Image
from torchvision import transforms

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Pad(padding=10),
        transforms.ToTensor()   
    ])
    image = preprocess(image)
    return image

def vectorize(images, fclip, save_folder):
    preprocess_images = [preprocess_image(img) for img in images]
    image_embeddings = fclip.encode_images(images, batch_size=32)
    image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)

    for img_path, embeddind in zip(images, image_embeddings):
        img_name = os.path.basename(img_path)
        vector_filename = os.path.splitext(img_name)[0] + '.npy'
        vector_path = os.path.join(save_folder, vector_filename)
        np.save(vector_path, vector_filename)