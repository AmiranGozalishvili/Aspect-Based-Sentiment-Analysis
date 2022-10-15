import warnings

import pandas as pd
import spacy
from tqdm.notebook import tqdm_notebook

from analysis import display_ngram_frequency
from datapreprocess import *
from opinionparse import opinion_parser
from predictpolarity import dependency_matching, polarity
from word_clouds import generate_wordcloud


def commands():
    # load a trained English pipeline
    nlp = spacy.load("en_core_web_sm")

    # initiate tqdm for pandas.apply() functions
    tqdm_notebook.pandas()

    # suppress all warnings
    warnings.simplefilter('ignore')

    # expand notebook display options for dataframes
    pd.set_option('display.max_colwidth', 200)
    pd.options.display.max_columns = 999
    pd.options.display.max_rows = 300

    # dataset = df

    # data preprocess
    dataset = raw_data[['Review Text', 'Recommended IND', 'Department Name']]

    # filter the dataset for our aspect
    dataset = dataset[(dataset['Review Text'].str.contains(r'colors?\b')) &
                      (dataset['Recommended IND'] == 0) &
                      (dataset['Department Name'] == 'Dresses')
                      ]

    print("Num. of observations:", len(dataset))

    # make all characters uniformly lowercase
    dataset['Review Text'] = dataset['Review Text'].apply(lambda x: x.lower())

    # expand contractions
    dataset['Review Text'] = dataset['Review Text'].apply(cont_expand)

    # clean slang
    dataset['Review Text'] = dataset['Review Text'].apply(clean_slang)

    # extract opinion

    """## Extract Opinion Units"""

    # (optional) load the already segmented opinions from the Datasets folder
    dataset = pd.read_excel("data/Saved_ASBA_Opinions.xlsx")

    # slice data to limit process time
    dataset = dataset[0:10]

    dataset["Opinion"] = dataset["Review Text"].progress_apply(opinion_parser)
    # print("opinion parser", dataset.head())

    # predict polarity
    dataset["Polarity"] = dataset["Opinion"].progress_apply(polarity)

    # dataset.head(1)

    dataset['Descriptors'] = dataset['Opinion'].progress_apply(dependency_matching)

    # dataset.sample(3)

    # Polarity Analysis
    # polarity = polarity()

    # Descriptor Analysis (n-gramming)

    positives = dataset[dataset["Polarity"] > 0]  # polarity greater than 0
    negatives = dataset[dataset["Polarity"] < 0]  # polarity less than 0

    # list all negative descriptors in a single string
    descriptors_negative_opinions = " ".join(negatives["Descriptors"].tolist())

    # positives
    descriptors_positive_opinions = " ".join(positives["Descriptors"].tolist())

    display_ngram_frequency(descriptors_negative_opinions, n=3, display=10)

    # wordclouds

    # WordCloud: descriptors extracted from negative opinions
    generate_wordcloud(descriptors_negative_opinions)

    # WordCloud: descriptors extracted from positive opinions
    generate_wordcloud(descriptors_positive_opinions)

    """## Examples"""

    opinion_texts = ["the color was beautiful",
                     "gorgeous colors",
                     "i ordered the red, it is a beautiful, vibrant, festive color",
                     "the colors were very bland and the flowers just hang",
                     "the color was not vibrant like photos show",
                     ]
    df_examples = pd.DataFrame(opinion_texts, columns=["Opinion"])
    df_examples["Polarity"] = df_examples["Opinion"].apply(polarity)  # polarity
    df_examples['Descriptors'] = df_examples['Opinion'].apply(dependency_matching)  # extract adjectives/adverbs

    return
