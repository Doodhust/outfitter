from fashion_clip.fashion_clip import FashionCLIP
from vectorization import vectorize, preprocess_image

import os

fclip = FashionCLIP('fashion-clip')

if __name__ == "__main__":
    img_path = '/home/kris/code/uni/outfitter/outfitter/src/db/images'
    vector_path = '/home/kris/code/uni/outfitter/outfitter/src/db/vectors'

    image_files = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith(('.jpeg', '.jpg', '.png'))]

    if not os.path.exists(vector_path):
        os.makedirs(vector_path)
    

    batch_size = 64

    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        vectorize(batch, fclip, vector_path)

    # vectorize(image_files, fclip, vector_path)