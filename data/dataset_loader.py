import os
import numpy as np
import pandas as pd
from PIL import Image

class Dataset_Loader:

    def __init__(self, dataset_path):
        
        self.rows = []
        self.columns = ['label'] + [f'pixel_{i}' for i in range(28*28)]
        
        for label in range(10):
            folder = os.path.join(dataset_path, str(label))
            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)
                img = Image.open(img_path).convert('L').resize((28, 28)) # -> convert to gray scale, resize all images to (28x28)
                pixels = np.array(img).flatten() / 255.0 # -> convert each image to pixels and flatten it into 1D numpy array
                row = np.concatenate(([label], pixels))
                self.rows.append(row)

        output_path = os.path.join(os.path.dirname(__file__), "digits_dataset.csv")
        df = pd.DataFrame(self.rows, columns=self.columns)
        df.to_csv(output_path, index=False)