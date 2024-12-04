import csv
import math
import os


class Save:
    def __init__(self, file_name="responses.csv"):
        self.file_name = file_name

    def save_to_csv(self, question, answer):
        file_exists = os.path.isfile(self.file_name)

        with open(self.file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Question", "Answer", "Score"])

            writer.writerow([question, answer, math.nan])
