from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datapreprocess import *

dataset = df

"""## Polarity Analysis"""


def polarity():
    print('polarity analysis checkpoint')

    ax = sns.histplot(dataset["Polarity"],
                      bins=np.arange(-1, 1, 0.2)
                      ).set_title('Histogram of Polarities (Aspect = "Color")')
    plt.show()

    # save the chart figure
    fig = ax.get_figure()
    fig.savefig('Polarity_Distribution.png',
                dpi=75,
                bbox_inches="tight")


"""## Descriptor Analysis (n-gramming)"""


def ngrams(text, n):
    return zip(*[text[i:] for i in range(n)])


def display_ngram_frequency(corpus, n, display):
    """
    Generate a DataFrame of n-grams and their frequencies.
    """
    ngram_counts = Counter(ngrams(corpus.split(), n))
    most_commmon = ngram_counts.most_common(display)

    ngram = []
    count = []
    for i in range(0, len(most_commmon)):
        ngram.append(" ".join(most_commmon[i][0]))
        count.append(most_commmon[i][1])

    if n > 3:
        col = f"{n}-gram"
    if n == 3:
        col = 'Tri-gram'
    if n == 2:
        col = 'Bi-gram'

    return pd.DataFrame(zip(ngram, count), columns=[col, "Count"])
