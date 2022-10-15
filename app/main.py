import pandas as pd
import spacy
import time
import warnings
from analysis import display_ngram_frequency
from datapreprocess import *
from fastapi import FastAPI
from monkeylearn import MonkeyLearn
from predictpolarity import dependency_matching, polarity
from tqdm.notebook import tqdm_notebook
from word_clouds import generate_wordcloud

app = FastAPI()

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
print("xlsx dataset", dataset.head(1))
dataset = dataset[0:10]
print(len(dataset))

# instantiate the client using your API key
ml = MonkeyLearn('2544e6156c96f1e16f10e3846604e78d62df5a7f')

# opinion unit extractor
model_id = 'ex_N4aFcea3'


def opinion_parser(text):
    """
    Extract the individual opinion unit (or phrase) within the text
    that contains the aspect term.
    """
    result = ml.extractors.extract(model_id, [text])
    time.sleep(1)

    extractions = result.body[0]['extractions']

    opinion_units = []
    num__opinion_units = len(extractions)

    for i in range(num__opinion_units):
        opinion_unit = "".join([extractions[i]['extracted_text']])

        if re.search("colors?", opinion_unit):
            return opinion_unit

    return ""


dataset["Opinion"] = dataset["Review Text"].progress_apply(opinion_parser)
print("opinion parser", dataset.head())

# predict polarity
dataset["Polarity"] = dataset["Opinion"].progress_apply(polarity)

dataset.head(1)

dataset['Descriptors'] = dataset['Opinion'].progress_apply(dependency_matching)

dataset.sample(3)

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


# 1 take text and apply (progress_apply(opinion_parser))  Review text to opinion
# 2 take opinion text and apply (progress_apply(polarity))  opinion to polarity
# 3 take opinion and apply (progress_apply(dependency_matching)) opinion to descriptors

@app.get('/')
def get_root():
    return {'message': 'Welcome to ASBA API'}


@app.get('/ASBA')
def textanalyze(text):
    Opinion_text = opinion_parser(text)
    Polarity = polarity(text)
    Descriptors = dependency_matching(text)
    return ("Opinion Text", Opinion_text), ("Polarity", Polarity), ("Descriptor", Descriptors)
