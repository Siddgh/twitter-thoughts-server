from fastapi import FastAPI, Response

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import twint
import nest_asyncio
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect


nest_asyncio.apply()


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained(
    'nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained(
    'nlptown/bert-base-multilingual-uncased-sentiment')


def get_tweets_by_keyword(keyword, fetch_rate=50):
    k = twint.Config()
    k.Pandas = True
    k.Lang = 'en'
    k.Search = keyword
    k.Limit = fetch_rate

    twint.run.Search(k)
    df = twint.storage.panda.Tweets_df
    return df


def get_users_tweets(username, fetch_rate=50):
    u = twint.Config()
    u.Pandas = True
    u.Lang = 'en'
    u.Search = 'from:@'+str(username)
    u.Limit = fetch_rate

    twint.run.Search(u)
    df = twint.storage.panda.Tweets_df
    return df


def clean_tweets(tweet):
    tweet = tweet.lower()
    tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)  # Remove Mentions
    tweet = re.sub("#[A-Za-z0-9_]+", "", tweet)  # Remove Hashtags
    tweet = re.sub(r"http\S+", "", tweet)  # Remove Hyper Links
    tweet = re.sub(r"www.\S+", "", tweet)  # Remove Hyper Links
    tweet = re.sub('[()!?]', '', tweet)  # Remove Punctuations
    tweet = re.sub('\[.*?\]', '', tweet)  # Remove Punctuations
    tweet = re.sub("[^a-z0-9]", " ", tweet)  # Remove Non-Alphanumeric Values
    tweet = tweet.lstrip().rstrip()
    tweet = ' '.join(tweet.split())
    return tweet


def get_sentiment_scores(tweet):
    tokens = tokenizer.encode(tweet, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1


def detect_lang(tweet):
    try:
        val = detect(tweet)
    except:
        val = 'nan'
    return val


@app.get('/keyword/{keyword}/{fetchrate}')
def trend_sentiments(keyword, fetchrate):
    keyword_tweets_df = get_tweets_by_keyword(keyword, fetchrate)
    if len(keyword_tweets_df) > 0:
        keyword_tweets_df['cleaned_tweets'] = keyword_tweets_df['tweet'].apply(
            clean_tweets)
        keyword_tweets_df['cleaned_tweets'] = keyword_tweets_df['cleaned_tweets'].replace(
            '', np.nan)
        keyword_tweets_df = keyword_tweets_df.dropna(
            axis=0, subset=['cleaned_tweets'])

        keyword_tweets_df['lang'] = keyword_tweets_df['cleaned_tweets'].apply(
            detect_lang)

        keyword_input_df = keyword_tweets_df.loc[keyword_tweets_df['lang'] == 'en']
        keyword_input_df['sentiment_scores'] = keyword_input_df['cleaned_tweets'].apply(
            get_sentiment_scores)

        res_json = keyword_input_df[[
            'cleaned_tweets', 'sentiment_scores']].to_json()

        print(len(keyword_tweets_df), len(keyword_input_df))
        return Response(content=res_json, media_type="application/json")
    else:
        return f'nan'


@app.get('/user/{keyword}/{fetchrate}')
def trend_sentiments(keyword, fetchrate):
    keyword_tweets_df = get_users_tweets(keyword, fetchrate)
    if len(keyword_tweets_df) > 0:
        keyword_tweets_df['cleaned_tweets'] = keyword_tweets_df['tweet'].apply(
            clean_tweets)
        keyword_tweets_df['cleaned_tweets'] = keyword_tweets_df['cleaned_tweets'].replace(
            '', np.nan)
        keyword_tweets_df = keyword_tweets_df.dropna(
            axis=0, subset=['cleaned_tweets'])

        keyword_tweets_df['lang'] = keyword_tweets_df['cleaned_tweets'].apply(
            detect_lang)

        keyword_input_df = keyword_tweets_df.loc[keyword_tweets_df['lang'] == 'en']
        keyword_input_df['sentiment_scores'] = keyword_input_df['cleaned_tweets'].apply(
            get_sentiment_scores)

        res_json = keyword_input_df[[
            'cleaned_tweets', 'sentiment_scores']].to_json()
        return Response(content=res_json, media_type="application/json")
    else:
        return f'nan'
