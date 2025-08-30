## CONFIG
# imports
import math
import numpy as np
import pandas as pd

import os
import re
from tqdm import tqdm

BASEPATH = f"/mnt/c/Workspace/Gerald/Programming/Hackathons/TikTok TechJam 2025" # configure this to match your own path
DATAPATH = f"{BASEPATH}/data"
# print(os.listdir(BASEPATH)) # this should display the correct folder if you have configured correctly

## LOAD DATASET

# load the dataset into a df
def _load_dataset(filename):
  """
  Takes filename of a .csv file containing the data
  Returns a pandas dataframe
  """
  df = pd.read_csv(f"{DATAPATH}/{filename}")
  return df # replace this

# testing
df = _load_dataset("scraped_reviews.csv")
# print(df.head())


## CLEAN DATA
# removing punctuation
# def strip_punctuation_df(df):
#     df['text'] = df['text'].map(lambda s: re.sub(r'[^\w\s]', '', s))
#     return df
def strip_punctuation(text):
  result = re.sub(r'[^\w\s]', '', text)
  return result

# # removing capitalisation
# def lower_text_df(df):
#     df['text'] = df['text'].map(lambda s: s.lower())
#     return df
def lower_text(text):
  return text.lower()

# def find_url(text):
#     url_pattern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
#     return re.search(url_pattern, text)

# def has_url_df(df):
#     return df.map(lambda x: True if find_url(x) else False)


## PROFANITY MODEL
from profanity_check import predict, predict_prob
class ProfanityFilter:
  """
  Pass in reference review data to sample and find the upper quartile threshold
  for offensiveness according to respective rating given

  Run fail() on new data to perform a simple test to see if it should be escalated based on a simple evaluation of its offensiveness relative to the rating given
  """

  def __init__(self, reference_data, text_col_name = 'Scraped Reviews', rating_col_name = 'Scraped Ratings'):
    reference_data['profanity_prob'] = predict_prob(reference_data[text_col_name])
    self.rating_map = {}
    for i in range(1, 6):
      rating_group = reference_data[reference_data[rating_col_name] == i]
      self.rating_map[i] = rating_group['profanity_prob'].quantile(q=0.75)

  def calc_prob(self, data, text_col_name='Scraped Reviews', rating_col_name='Scraped Ratings'):
    data['profanity_prob'] = predict_prob(data[text_col_name])
    return data

  def fail(self, text, rating):
    return predict_prob([text])[0] > self.rating_map[rating]

  def fail_df(self, data, text_col_name='Scraped Reviews', rating_col_name='Scraped Ratings'):
    data = self.calc_prob(data)
    return data.apply(lambda x: x['profanity_prob'] > self.rating_map[x[rating_col_name]], axis=1)
    # return data.apply(lambda x: x[text_col_name])
    # data.apply(lambda x: print(x[text_col_name]), axis=1)

  def ok(self, text, rating, **kwargs):
    return not self.fail(text, rating)

  def ok_df(self, data, text_col_name='Scraped Reviews', rating_col_name='Scraped Ratings'):
    data = self.calc_prob(data)
    return data.apply(lambda x: x['profanity_prob'] <= self.rating_map[x[rating_col_name]], axis=1)
  
prof_filter = ProfanityFilter(df)

# print(prof_filter.ok("Fuck you", 5))
# print(prof_filter.ok("you are very nice and cool", 1))


# SECONDHAND REVIEWS MODEL
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter
from torch import nn

# Config
seed = 42
torch.manual_seed(seed)

# Load Dataset
secondhand_df = _load_dataset("secondhand_reviews_labelled.csv")
secondhand_df.head()

# Simple encoder
def get_corpus(df, top_n=None):
  all_words = []
  for i, row in df.iterrows():
    all_words.extend([word for word in row.text.split()])
  count_words = Counter(all_words)
  return count_words.most_common(top_n)

corpus = get_corpus(secondhand_df, top_n=10000)

def get_encodings(corpus):
  # simple encoder for now which just returns a number
  mapping = {w:i+1 for i, (w, c) in enumerate(corpus)}
  return mapping

word2int = get_encodings(corpus)

def encode(text, max_length=128):
  encoding = np.zeros((1, max_length), dtype = int)
  for i, w in enumerate(text.split()[:max_length]):
    encoding[0][i] = word2int[w] if w in word2int else 0

  return encoding

# print(encode("ive never been here but a friend of a friend said the parking is impossible so one star"))

# Classifier Model
EMBED_DIM = 256
NUM_ENCODER_LAYERS = 1
NUM_HEADS = 2

class EncoderClassifier(nn.Module):
  def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
    super(EncoderClassifier, self).__init__()
    self.emb = nn.Embedding(vocab_size, embed_dim)
    self.encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        batch_first=True
    )
    self.encoder = nn.TransformerEncoder(
        encoder_layer = self.encoder_layer,
        num_layers=num_layers
    )
    self.linear = nn.Linear(embed_dim, 1)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = self.emb(x)
    x = self.encoder(x)
    x = self.dropout(x)
    x = x.max(dim = 1)[0]
    out = self.linear(x)
    return out
  
# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EncoderClassifier(
    # len(word2int) + 1,
    912,
    embed_dim=EMBED_DIM,
    num_layers=NUM_ENCODER_LAYERS,
    num_heads=NUM_HEADS
).to(device)

# print(model)
# Total parameters and trainable parameters.
# total_params = sum(p.numel() for p in model.parameters())
# print(f"{total_params:,} total parameters.")
# total_trainable_params = sum(
#     p.numel() for p in model.parameters() if p.requires_grad)
# print(f"{total_trainable_params:,} training parameters.\n")

# TRAINING
def count_correct_incorrect(labels, outputs, train_running_correct):
    # As the outputs are currently logits.
    outputs = torch.sigmoid(outputs)
    running_correct = 0
    for i, label in enumerate(labels):
        if label < 0.5 and outputs[i] < 0.5:
            running_correct += 1
        elif label >= 0.5 and outputs[i] >= 0.5:
            running_correct += 1
    return running_correct

# Training Function
def train(model, df, optimizer, loss_fn, device):
  """
  train one epoch
  """
  model.train()
  train_running_loss = 0.0
  train_running_correct = 0
  counter = 0
  for i, row in tqdm(df.iterrows()):
    counter += 1

    X, y = row['text'], row['secondhand']
    inputs = torch.tensor(encode(X), dtype=torch.int32).to(device)
    labels = torch.tensor([y], dtype=torch.float32).to(device)
    optimizer.zero_grad()

    outputs = model(inputs)
    outputs = torch.squeeze(outputs, -1)

    loss = loss_fn(outputs, labels)
    train_running_loss += loss.item()
    running_correct = count_correct_incorrect(
        labels, outputs, train_running_correct
    )
    train_running_correct += running_correct
    loss.backward()
    optimizer.step()

  epoch_loss = train_running_loss / counter
  epoch_acc = 100. * (train_running_correct / len(df))
  return epoch_loss, epoch_acc

# DO NOT RUN FOR FINAL
# Training Parameters
# BATCH_SIZE = 32
# VALID_SPLIT = 0.20
# EPOCHS = 3
# LR = 0.00001

# Config
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(
#     model.parameters(),
#     lr=LR
# )

# train_df = secondhand_df.sample(frac=0.8)
# test_df = secondhand_df.drop(train_df.index)
# print(train_df.head(), test_df.head())
# print(len(train_df), len(test_df))

# Training Loop
# for epoch in range(EPOCHS):
#   print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
#   epoch_loss, epoch_acc = train(model, train_df, optimizer, loss_fn, device)
#   print(epoch_loss, epoch_acc)

model.load_state_dict(torch.load(f"{BASEPATH}/models/secondhand_review_model", weights_only=True))
model.eval()

# Testing model output
# text = "ive never been here but a friend of a friend said the parking is impossible so one star"
# text = "the staff was a bit unhelpful when i asked for advice on a new hairstyle"
# text = "the massage therapist was incredible the pressure was just right and i felt so relaxed afterwards"
# text = "never been here but somebody told me the food is bad so im giving it one star"
# example = torch.tensor(encode(text), dtype=torch.int32).to(device)
# pred = model(example)
# pred = torch.squeeze(pred, -1)
# pred = torch.sigmoid(pred)
# print(pred)
# print(pred.item(), (pred > 0.5).item())

# Testing model output
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# counter = 0
# for i, row in test_df.iterrows():
#     text, label = row['text'], row['secondhand']
#     inputs = torch.tensor(encode(text), dtype=torch.int32).to(device)
#     labels = torch.tensor([label], dtype=torch.float32).to(device)
#     outputs = model(inputs)
#     outputs = torch.squeeze(outputs, -1)
#     outputs = torch.sigmoid(outputs)

#     l = label
#     o = outputs.item()
#     if l < 0.5 and o < 0.5:
#         tn += 1
#     elif l >= 0.5 and o >= 0.5:
#         tp += 1
#     elif l < 0.5 and o >= 0.5:
#         fp += 1
#     else:
#         fn += 1

# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1_score = 2 / ((1 / precision) + (1 / recall))

# print(tp, tn, fp, fn)
# print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}")

def secondhand_filter(text, **kwargs):
    inputs = torch.tensor(encode(text), dtype=torch.int32).to(device)
    pred = model(inputs)
    pred = torch.squeeze(pred, -1)
    pred = torch.sigmoid(pred)
    return not (pred > 0.5).item()


## AD DETECTION
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Config
import joblib
filename = "ad_detection.joblib"
filepath = f"{BASEPATH}/models/{filename}"

loaded_ad_model = joblib.load(filepath);

msg1 = "You have won $100000 prize! Contact us for the reward!"
msg2 = "Best pizza! Visit www.pizzapromo.com for discounts!"
msg3 = "Food is good.we ordered Kodi drumsticks and basket mutton biryani. All are good. Thanks to Pradeep. He served well. We enjoyed here. Ambience is also very good."

# print(loaded_ad_model.predict([msg1]))
# print(loaded_ad_model.predict([msg2]))
# print(loaded_ad_model.predict([msg3]))

def ad_filter(text, **kwargs):
  result = loaded_ad_model.predict([text])[0]
  if result == "ad":
    return False
  return True

### RELEVANCY
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = ["highly relevant", "moderately relevant", "slightly relevant", "irrelevant"]

def get_relevance_data(review_text, location_context):
    if pd.isna(review_text) or not str(review_text).strip():
        return {'label': 'irrelevant', 'score': 0.0}

    user_query = f"Based on the context of {location_context}, which reviews regardless of sentiment, should I consider when visiting this location to engage in their services?"

    #f"Based on the context of {location_context}, which reviews would be the most relevant to a user that is considering to visit {location_context}, irregardless of sentiment and tone?"

    #f"Based on the context of {location_context}, which reviews would be the most relevant to a user that is considering to engage with this business?"

    result = classifier(review_text, [user_query] + candidate_labels) 

    relevance_score = result['scores'][0]

    if relevance_score >= 0.85:
        relevance_label = "highly relevant"
    elif relevance_score >= 0.55:
        relevance_label = "moderately relevant"
    elif relevance_score >= 0.25:
        relevance_label = "slightly relevant"
    else:
        relevance_label = "irrelevant"

    return {'label': relevance_label, 'score': relevance_score}

# print(get_relevance_data("Order first round of fries is soggy, it was replaced and 2nd round is so oily that it’s glowing with oil! First time I see mac Donald’s fries shinning with oily surfaces. To me, this is disgusting and sloppy work. Why not drain away the oil properly before serving???", "McDonald's"))

def relevance_filter(text, location, **kwargs):
  result = get_relevance_data(text, location)
  return not result['score'] < 0.48

## FINAL PIPELINE
class Pipeline:
  def __init__(self):
    self.simple_filters = {}
    self.preprocessing_steps = []
    self.model_filters = {}

  def register_simple_filter(self, filter, name):
    """
    Pass in a function that takes in text and optional additional data params
    The function should return True if the text is ok,
    and False if it should be flagged for further review
    """
    self.simple_filters[name] = filter

  def simple_filter(self, text, data):
    for name, filter in self.simple_filters.items():
      if not filter(text, **data):
        print(f"Error: {name} violation")
        return False
    return True

  def register_preprocessing_step(self, step):
    self.preprocessing_steps.append(step)

  def preprocess(self, text):
    for step in self.preprocessing_steps:
      text = step(text)
    return text
  
  def register_model_filter(self, filter, name):
    """
    Pass in a function that takes in text and optional additional data params
    The function should return True if the text is ok,
    and False if it should be flagged for further review
    """
    self.model_filters[name] = filter

  def model_filter(self, text, data):
    for name, filter in self.model_filters.items():
      if not filter(text, **data):
        print(f"Error: {name} violation")
        return False
    return True

  def evaluate(self, text, data):
    if not self.simple_filter(text, data):
      return False #lgtm! likely doesn't need further more expensive checks
    text = self.preprocess(text)
    #insert more expensive models
    if not self.model_filter(text, data):
      return False


pipeline = Pipeline()
pipeline.register_simple_filter(prof_filter.ok, "Profanity")
# print(pipeline.simple_filter('very good!', {'rating': 1}))
# print(pipeline.simple_filter('fuck you', {'rating': 5}))

pipeline.register_preprocessing_step(strip_punctuation)
pipeline.register_preprocessing_step(lower_text)

pipeline.register_model_filter(ad_filter, "Spam")
# print(pipeline.model_filter('You have won $100000 prize! Contact us for the reward!', {'rating': 5}))

pipeline.register_model_filter(relevance_filter, "Relevance")
# print(pipeline.model_filter('i just bought a new phone', {'rating': 5, 'location': "McDonald's"}))

pipeline.register_model_filter(secondhand_filter, "Secondhand")
# print(pipeline.model_filter('ive never been here but a friend of a friend said the parking is impossible so one star', {'rating': 5}))





























# DEMO AREA
ts = ["u stupid bad poop fart stupid dumb crap crappy", "Supertree Grove is a marvel ...", "never been here but somebody told me the food is bad so im giving it one star"]
ds = [{'rating': 5, 'location': "McDonald's"}, {'rating': 1, 'location': "McDonald's"}, {'rating': 1, 'location': "McDonald's"}]

for t, d in zip(ts, ds):
    print(f"text: {t}, rating: {d['rating']}, location: {d['location']} | ", end="")
    pipeline.evaluate(t, d)