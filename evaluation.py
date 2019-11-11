import csv
import os
import numpy as np


class evaluation:
    def __init__(self, dest_folder="graphs"):
        self.dest_folder = dest_folder
        self.metrics_keys = []
        self.metrics_values = []
    
    def set_keys(self, keys):
        self.metrics_keys = keys
        
    def add_value(self, values):
        self.metrics_values.append(values)

    def save_as_csv(self, filename):
        with open(os.path.join(self.dest_folder, filename), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metrics_keys)
            writer.writeheader()
            for data in self.metrics_values:
                writer.writerow(data)