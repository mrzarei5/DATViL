import os

from .utils import DatasetBase, read_json
from .oxford_pets import OxfordPets


template = ['a photo of {}, a type of food.']


class Food101(DatasetBase):

    dataset_dir = 'food-101'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Food101.json')
        self.attribute_path = os.path.join(root,'attributes_gpt4','food101.json')
        
        self.template = template
        
        attributes = read_json(self.attribute_path)
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test, attributes = attributes)