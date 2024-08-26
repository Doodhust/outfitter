from fashion_clip.fashion_clip import FashionCLIP
import os
from vectorization import vectorize

fclip = FashionCLIP('fashion-clip')

img_path = '/home/doodhust/sourse/new_summer_project_DNS/data/clothes' # 2 фотографии на вход
vector_path = 'datavectors/clothes' # 2 файла npy на выход

image_files = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith(('.jpeg', '.jpg', '.png'))]

if not os.path.exists(vector_path):
    os.makedirs(vector_path)

vectorize(image_files, fclip, vector_path)