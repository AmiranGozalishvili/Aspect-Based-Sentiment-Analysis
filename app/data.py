import pandas as pd

def load_data():
    df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")

    return df
