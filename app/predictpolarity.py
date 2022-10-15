import spacy
from textblob import TextBlob

from datapreprocess import *

# load a trained English pipeline
nlp = spacy.load("en_core_web_sm")


## python -m spacy download en_core_web_lg
## python -m spacy download en_core_web_sm


def polarity(text):
    """
    Predict the polarity of the text using TextBlob.
    Results range from negative to positive on a scale of [-1, +1].
    """
    testimonial = TextBlob(text)
    return round(testimonial.sentiment.polarity, 2)


"""## Identify Descriptors"""


def dependency_matching(text):
    """
    Identify and extract word(s) that are describing
    the aspect term.
    """
    doc = nlp(text)

    tags = ['JJ', 'JJR', 'JJS']
    dependents = ['acomp', 'advmod']

    extraction = []
    for i, token in enumerate(doc):

        # location of color in sequence
        if re.search("color", token.text):
            color_pos = i

        if ((token.dep_ in dependents) or (token.tag_ in tags)) and re.search("color", token.head.text):
            extraction.append(token.text)

        if token.dep_ == 'acomp':
            extraction.append(token.text)

            children = [child for child in token.children]
            if len(children) > 0 and str(children[0]).isalpha():
                extraction.insert(0, str(children[0]))

            for t in range(4):
                try:
                    if doc[i - t].dep_ == 'neg':
                        negation = doc[i - t].text
                        extraction.insert(0, negation)
                except:
                    continue

        # look for adjectives near the aspect if no matches were found yet
        if len(extraction) == 0 and i == len(doc) - 1:
            for t in range(-6, 6):
                try:
                    if doc[color_pos + t].tag_ in tags:
                        if doc[color_pos + t].text in extraction:
                            continue
                        extraction.append(doc[color_pos + t].text)

                        children = [child for child in doc[color_pos + t].children]
                        if len(children) > 0 and str(children[0]).isalpha():
                            extraction.insert(0, str(children[0]))
                except:
                    continue

    return " ".join(extraction)
