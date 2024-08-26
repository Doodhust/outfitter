from fashion_clip.fashion_clip import FashionCLIP
import os
from vectorization import vectorize

fclip = FashionCLIP('fashion-clip')

if __name__ == "__main__":
    # img_path = 'C:/home/code/Stylematch AI/scripts/data/fashiers'
    # vector_path = 'C:/home/code/Stylematch AI/scripts/data/vectors/fashiers'
    img_path = '/home/doodhust/sourse/new_summer_project_DNS/data/test'
    vector_path = 'datavectors/test_1'
    


    image_files = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith(('.jpeg', '.jpg', '.png'))]

    if not os.path.exists(vector_path):
        os.makedirs(vector_path)

    vectorize(image_files, fclip, vector_path)


