# MIT License
#
# Copyright (c) 2023 Panagiotis Anagnostou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
    This script reads the dataset and splits it into train and test sets.
    The default split is 80% train and 20% test. The script also has the option
    to clean the text of the abstracts. The cleaning process is described in the
    clean_text function and is based on the cleaning process of the
    dataset. The data are separated by the title of the paper, so that the same
    paper is original and generated abstracts to be in the same set. Finally,
    the train and test sets are saved as csv files in the same directory, with
    the name "train.csv" and "test.csv" respectively.
"""

from tqdm import tqdm

import pandas as pd
import re
import stopwordsiso as stopwords
import sys

import nltk
from nltk.tokenize import sent_tokenize


def clean_text(essay):
    """
    Function for basic text cleaning of the dataset. The cleaning we
    followed steps are:

    1. Remove HTML tags
    2. Remove -
    3. Remove punctuation
    4. Convert to lowercase
    5. Remove numbers
    6. Remove stop words
    7. Remove extra whitespace
    8. Remove study and paper keywords. These keywords are not useful for the
    classification task, because we classify scientific papers.

    Parameters
    ----------
    abstract : str
        The text to be cleaned.

    Returns
    -------
    text : str
        The cleaned text.

    """
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", essay)

    # Remove -
    # text = re.sub(r"-", " ", text)
    text = re.sub(r"\u2013", " ", text)

    # Remove punctuation
    # text = re.sub(r"[^\w\s]", "", text)

    # Convert to lowercase
    # text = text.lower()

    # Remove numbers
    # text = re.sub(r"\d", "", text)

    # Remove stop words
    # text = " ".join(
    #     [word for word in text.split() if word not in stopwords.stopwords("en")]
    # )

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # # Split long text
    # if len(text) > 15000:
    #     text, part2 = split_long_text(text)
    #     # print("splited")

    # if len(text) < 3000:
    #     print("short")
    

    return text


def split_long_text(text):
    sentences = sent_tokenize(text)

    mid_index = len(sentences) // 2

    part1 = ' '.join(sentences[:mid_index])
    part2 = ' '.join(sentences[mid_index:])

    return part1, part2


if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv("datasets/wiki.csv", engine="c").tail(300)

    # Initialize the command line arguments
    clean = "true"
    split = 1
    random_state = 42
    # Acceptable command line arguments
    command_line_args = ["clean", "split", "random_state"]

    # Check if the user provided any command line arguments and if they have valid names
    if len(sys.argv) > 4:
        raise ValueError("Too many arguments")
    elif len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.split("=")[0] not in command_line_args:
                raise ValueError(f"Invalid argument: {arg.split('=')[0]}")
            else:
                if arg.split("=")[0] == "clean":
                    clean = arg.split("=")[1]
                elif arg.split("=")[0] == "split":
                    split = arg.split("=")[1]
                elif arg.split("=")[0] == "random_state":
                    random_state = arg.split("=")[1]

    # Check if the values of the clean, split and random_state command line arguments are valid
    try:
        split = float(split)
    except ValueError:
        raise ValueError("split: Invalid split value")
    if split < 0 or split > 1:
        raise ValueError("split: Invalid split value")

    if clean:
        if clean.lower() == "true":
            clean = True
        elif clean.lower() == "false":
            clean = False
        else:
            raise ValueError("clean: Invalid clean value")

    try:
        random_state = int(random_state)
    except ValueError:
        raise ValueError("random_state: Invalid random_state value")
    if random_state < 0:
        raise ValueError("random_state: Invalid random_state value")

    # Clean the text if the user provided the clean command line argument
    if clean:
        print("Cleaning text...")
        tqdm.pandas()
        # df["essay"] = df["essay"].progress_apply(clean_text)
        df["wiki_intro"] = df["wiki_intro"].progress_apply(clean_text)
        df["generated_intro"] = df["generated_intro"].progress_apply(clean_text)

    percent = split
    print("Splitting data into train and test sets...")
    # print(
    #     f"The train set contains {split * 100} % and the test set contains {100 - split * 100} % of the data"
    # )

    # Apply the train split to the original and generated abstracts
    training_df = df.head(100)#.sample(
        # frac=percent, random_state=random_state).reset_index(drop=True)
    validation_df = df.tail(10)

    print(f"The train set shape is {training_df.shape}")
    print(f"The test set shape is {validation_df.shape}")

    # Save the train and test sets as csv fil
    print("Saving train and test sets...")
    validation_df.to_csv("test.csv", index=False)
    training_df.to_csv("training.csv", index=False)
    print("Done!!!")
