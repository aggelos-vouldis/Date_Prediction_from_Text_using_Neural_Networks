import os

import pandas as pd
import json


class Import:
    def __init__(self) -> None:
        pass

    class data:
        def __init__(self, file_path) -> None:
            self.file_path = file_path
            pass

        def from_csv(self, delim: str = "\t", low_memory: bool = False, encoding: str = 'utf-8') -> pd.DataFrame:
            """Function to get data from CSV file with no low memory and encoding = 'utf-8'"""
            return pd.read_csv(self.file_path, delimiter=delim, low_memory=low_memory, encoding='utf-8')

        def from_json(self, encoding: str) -> json:
            with open(self.file_path, encoding='utf-8') as f:
                stop_words = json.load(f)

            return stop_words


class Export:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        pass

    def write_to_txt(self, message: str) -> None:
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(message)

    def append_to_txt(self, message: str) -> None:
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(message)


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
