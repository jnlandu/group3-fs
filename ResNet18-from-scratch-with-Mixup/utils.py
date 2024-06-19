import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from data import get_mappings, mapping_path


def display_images(path, df,classes, rows, cols):
    class_mapping_dict_number = get_mappings(mapping_path)

    for i in np.random.randint(1000, size=classes):

        number = rows * cols

        new_df = df[df['label'] == i]

        img_list = random.sample(new_df['image_id'].tolist(), number)

        plt.figure(figsize=(8, 3))
        for index, img_id in enumerate(img_list):
            plt.subplot(rows, cols, index+1)
            image = Image.open(path + '/' + img_id)
            plt.imshow(image, aspect='auto')
            plt.axis('off')

        plt.suptitle(f'\n\n Class {i}: ' + class_mapping_dict_number[i], fontsize=16)
        plt.tight_layout()




#count trainable parameters of the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)