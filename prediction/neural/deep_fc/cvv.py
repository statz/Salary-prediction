import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
train_data = pd.read_csv(data_path + "down_x_train_normalized.csv")
text = train_data["combined_text"].values.astype('U')
tfidf = CountVectorizer(min_df= 100 / len(text), binary=True, ngram_range=(1, 1),
                        stop_words=stopwords.words('russian') + stopwords.words('english'))
vect_text_train = tfidf.fit(text)
print(len(tfidf.vocabulary_))
