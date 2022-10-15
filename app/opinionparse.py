import time
import re

from monkeylearn import MonkeyLearn

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
