import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords, twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string
import matplotlib.pyplot as plt
import seaborn as sns

# from wordcloud import WordCloud

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                           reduce_len=True)
punctuations = string.punctuation
stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()


def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


def process_tweet(tweet):
    processed_tweet = remove_hyperlinks_marks_styles(tweet)
    tweet_tokens = tokenize_tweet(processed_tweet)
    tweets_clean = remove_stopwords_punctuations(tweet_tokens)
    tweets_stem = get_stem(tweets_clean)

    print(tweets_stem)
    return tweets_stem


def get_stem(tweets_clean):
    tweets_stem = []

    for word in tweets_clean:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)

    return tweets_stem


def tokenize_tweet(tweet):
    tweet_tokens = tokenizer.tokenize(tweet)

    return tweet_tokens


def remove_hyperlinks_marks_styles(tweet):
    # enlever les RT ancient tweeter
    new_tweet = re.sub(r'^RT[\s]+', '', tweet)
    # enlever les lien
    new_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', new_tweet)
    # hashtag
    # new_tweet = re.sub(r'#', '', new_tweet)

    return new_tweet


def remove_stopwords_punctuations(tweet_tokens):
    tweets_clean = []

    for word in tweet_tokens:
        if word not in stopwords_english and word not in punctuations:
            tweets_clean.append(word)

    return tweets_clean


def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"


PATH = 'C:/Users/LM/Downloads/dataset/test.csv'
TRAINING_PATH = 'C:/Users/LM/Downloads/dataset/train.csv'

Test_df = pd.read_csv(PATH)
Training_df = pd.read_csv(TRAINING_PATH, encoding='latin-1')
combine = Training_df.append(Test_df, ignore_index=True)  # train and test dataset are combined
print(combine
      )
length_train_dataset = Training_df['tweet'].str.len()
length_test_dataset = Test_df['tweet'].str.len()
plt.hist(length_train_dataset, bins=20, label="Train tweets")
plt.hist(length_test_dataset, bins=20, label="Test tweets")
plt.legend()
plt.show()

combine['clean_tweet'] = combine['tweet'].str.replace("[^a-zA-Z#]", " ")
combine['clean_tweet'] = combine['clean_tweet'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

tokenized_tweets = combine['clean_tweet'].apply(process_tweet)
for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
combine['clean_tweet'] = tokenized_tweets

ht_regular = hashtag_extract(combine['clean_tweet'][combine['label'] == 0])
ht_negative = hashtag_extract(combine['clean_tweet'][combine['label'] == 1])
ht_regular = sum(ht_regular, [])
ht_negative = sum(ht_negative, [])

nonracist_tweets = nltk.FreqDist(ht_regular)
df1 = pd.DataFrame({'Hashtag': list(nonracist_tweets.keys()), 'Count': list(nonracist_tweets.values())})

df1 = df1.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=df1, x="Hashtag", y="Count")
ax.set(ylabel="Count")
plt.show()

racist_tweets = nltk.FreqDist(ht_negative)
df2 = pd.DataFrame({'Hashtag': list(racist_tweets.keys()), 'Count': list(racist_tweets.values())})

df2 = df2.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=df2, x="Hashtag", y="Count")
plt.show()
combine = combine.fillna(0)

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combine['clean_tweet'])  # tokenize and build vocabulary

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combine['clean_tweet'])

X_train, X_test, y_train, y_test = train_test_split(tfidf, combine['label'],
                                                    test_size=0.2, random_state=69)
print("X_train_shape : ", X_train.shape)
print("X_test_shape : ", X_test.shape)
print("y_train_shape : ", y_train.shape)
print("y_test_shape : ", y_test.shape)

model_naive = MultinomialNB().fit(X_train, y_train)
predicted_naive = model_naive.predict(X_test)

score_naive = accuracy_score(predicted_naive, y_test)
print("Accuracy with Naive-bayes: ", score_naive)

#BAG-OF-WORD : 0.938873067534581
#TF-IDF : 0.9546379170056957