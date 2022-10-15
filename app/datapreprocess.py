from data import load_data
import contractions
import re


df = load_data()

raw_data = df
def datapreprocess(raw_data):
    # drop any rows with missing Review Text
    raw_data.dropna(axis=0,
                    how='any',
                    subset=['Review Text'],
                    inplace=True)


    # dataset = raw_data[['Review Text', 'Recommended IND', 'Department Name']]
    #
    #
    # # filter the dataset for our aspect
    # dataset = dataset[(dataset['Review Text'].str.contains(r'colors?\b')) &
    #                   (dataset['Recommended IND']==0) &
    #                   (dataset['Department Name']=='Dresses')
    #                   ]
    #
    # print("Num. of observations:", len(dataset))
    #


def clean_slang(text):
    """
    Correct common slang and abbreviations
    """
    text = re.sub(r"\babt?\b", "about", text)
    text = re.sub(r"\brll?y\b", "really", text)
    text = re.sub(r"\bso{2,}\b", "so", text)
    text = re.sub(r"\bmed\b", "medium", text)
    text = re.sub(r"\bxx?s\b", "extra small", text)
    text = re.sub(r"\bxx?l\b", "extra large", text)
    text = re.sub(r"\bfab\b", "fabulous", text)
    text = re.sub(r"\bblk\b", "black", text)
    text = re.sub(r"\bpromo\b", "promotion", text)
    text = re.sub(r"\btts\b", "true to size", text)
    text = re.sub(r"\blbs?\b", "pounds", text)
    text = re.sub(r"\brn\b", "right now", text)
    text = re.sub(r"\bwanna\b", "want to", text)
    text = re.sub(r"\besp\b", "especially", text)
    text = re.sub(r"\bgonn[ao]\b", "going to", text)
    text = re.sub(r"\btho\b", "though", text)
    text = re.sub(r"altho ", "although ", text)
    text = re.sub(r"prolly", "probably", text)
    text = re.sub(r"asap", "as soon as possible", text)
    text = re.sub(r"\bbc|b/c\b", "because", text)
    text = re.sub(r"\bavail\b", "available", text)
    text = re.sub(r"\bdiff\b", "different", text)
    text = re.sub(r"\bnxt|enxt\b", "next", text)
    text = re.sub(r" w/ ", " with ", text)
    text = re.sub(r"\bdidn ", "didn't ", text)
    text = re.sub(r" dnt ", " don't ", text)
    text = re.sub(r"\bsnd\b", "send", text)
    text = re.sub(r"b4\b", "before", text)
    text = re.sub(r"\batm\b", "at the moment", text)
    text = re.sub(r"\besp\b", "especially", text)
    text = re.sub(r"\bavg\b", "average", text)
    text = re.sub(r"\.{2,}", ". ", text)
    text = re.sub(r"[\\-]", ' ', text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def cont_expand(text):
    """
    Expand contractions (i.e don't->do not, can't->cannot)
    """
    expanded_text = contractions.fix(text, slang=False)
    return expanded_text



