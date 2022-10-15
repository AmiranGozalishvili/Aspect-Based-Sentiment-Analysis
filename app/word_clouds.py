from wordcloud import WordCloud

from analysis import *


def generate_wordcloud(text):
    """
    Generate word cloud images.
    """
    wordcloud = WordCloud(collocations=False,
                          background_color="black",
                          max_words=50).generate(text)

    # set the figure size
    plt.figure(figsize=[8, 10])

    # plot the wordcloud
    plt.imshow(wordcloud, interpolation="bilinear")

    # remove plot axes
    plt.axis("off")
