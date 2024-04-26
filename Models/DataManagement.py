from typing import List, Tuple

from pandas import DataFrame

from .Errors import ImportNotFoundException
import numpy as np
import pandas as pd


class ClearData:
    def __init__(self) -> None:
        ...

    def from_characters(self, dataframe: pd.DataFrame, r_characters: List[str], character: str,
                        column_names: List[str]) -> DataFrame:
        """Function to replace a number of characters (r_characters) from some columns of a dataframe
        with a character (character)"""
        if not (r_characters or column_names):  # check is lists are empty
            return dataframe

        dataframe[column_names] = dataframe[column_names].replace(r_characters, character, regex=True)

    def from_words(self, column: pd.Series, words_to_remove: List[str]) -> None:
        try:
            import nltk
            nltk.download('punkt')
        except ImportNotFoundException:
            raise ImportNotFoundException(message="Nltk wasn't found on your computer")

        for idx, sentence in enumerate(column):
            words: List[str] = nltk.tokenize.word_tokenize(sentence)
            words = [word for word in words if word not in words_to_remove]

            column[idx] = ' '.join(words)

    def from_empty_string_cells(self, dataframe: pd.DataFrame, column: str) -> None:
        dataframe[column].replace('', np.nan, inplace=True)
        dataframe.dropna(subset=[column], inplace=True)

    def from_whitespace(self, dataframe: pd.DataFrame, column_names: List[str]) -> None:
        for column in column_names:
            dataframe[column] = dataframe[column].str.strip()

    def from_single_characters(self, dataframe: pd.DataFrame, column_names: List[str]) -> None:
        dataframe[column_names] = dataframe[column_names].replace(r'\b\w\b', '', regex=True)

    def from_double_character_words(self, dataframe: pd.DataFrame, column_names: List[str]) -> None:
        dataframe[column_names] = dataframe[column_names].replace(r'\b\w\w\b', '', regex=True)

    def from_columns(self, dataframe: pd.DataFrame, column_names: List[str]) -> None:
        dataframe.drop(column_names, axis=1, inplace=True)

    def from_uppercase_letters(self, dataframe: pd.DataFrame, column_names: List[str]) -> None:
        for column_name in column_names:
            dataframe[column_name] = dataframe[column_name].str.lower()


class Vectorizer:
    def __init__(self) -> None:
        pass

    def tf_idf(self, list_of_strings) -> List[List[int]]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportNotFoundException:
            raise ImportNotFoundException(message="Sklearn wasn't found on your computer")

        return TfidfVectorizer().fit_transform(list_of_strings).toarray()


class Visualization:
    def __init__(self) -> None:
        ...

    def get_dictionary(self, column: pd.Series, n_items_returned: int = None) -> List[Tuple[str, int]]:
        """Function to get all the words in all texts with their corresponding frequency
    Leave n_items_returned variable empty for all the words to be returned"""
        try:
            import nltk
            nltk.download('punkt')
        except ImportNotFoundException:
            raise ImportNotFoundException(message="Nltk wasn't found on your computer")
        all_text: str = " ".join(column)
        all_words = nltk.tokenize.word_tokenize(all_text)
        all_words_dist = nltk.FreqDist(w.lower() for w in all_words)

        if n_items_returned is None:
            return all_words_dist.most_common()
        return all_words_dist.most_common()[-n_items_returned:]


class Preprocess:
    def __init__(self) -> None:
        pass

    def MinMaxScaler(self, data) -> None:
        try:
            from sklearn.preprocessing import MinMaxScaler
        except ImportNotFoundException:
            raise NotImplementedError("Sklearn is not downloaded properly")
        return MinMaxScaler().fit_transform(data)

    def slice_from_end_2d_array(self, array: np.array, n_slice: int) -> Tuple[np.array, np.array]:
        """Function to slice an array on 2 other from the end of the first one"""
        return array[:, :-n_slice], array[:, -n_slice:]

    def split_column_to_words(self, column: pd.Series) -> List[List[str]]:
        try:
            import nltk
            nltk.download('punkt')
        except ImportNotFoundException:
            raise ImportNotFoundException(message="Nltk wasn't found on your computer")

        words: List[List[str]] = list()
        for sentence in column:
            words.append(nltk.tokenize.word_tokenize(sentence))
        return words
