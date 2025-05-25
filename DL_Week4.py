# %% [markdown]
# # Week 4 Assignment: NLP Disaster Tweets Kaggle Mini-Project
# 
# The goal of this project is to classify tweets as either disaster-related or non-disaster-related. This will be achieved using deep learning Recurrent Neural Networks (RNNs) and Natural Language Processing (NLP) techniques. We will be working with the ["Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started/data) dataset from Kaggle.  The project involves converting tweets into a format understandable by our model, training a series of RNNs to identify the best architecture for this classification task, and comparing its performance against several supervised learning models. Model performance will be evaluated using metrics such as accuracy and F1-score, the latter being the primary metric as it is the standard for judging performance in the Kaggle competition.
# 

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter

# Deep Learning Model Layes
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding,GRU, Dense
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Conv1D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Supervised Learning Models
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Metrics
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from tensorflow.keras.metrics import F1Score

# Preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# NLP
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions
from wordcloud import WordCloud, STOPWORDS

# Set up environment (for Colab)
tf.compat.v1.enable_eager_execution()
tf.data.experimental.enable_debug_mode()

pd.set_option("display.max_colwidth", 150)

#print(tf.config.list_physical_devices())

# %% [markdown]
# # Data Inspection
# 
# We began by loading the Kaggle "Natural Language Processing with Disaster Tweets" dataset. The training set contains 7,613 tweets, each labeled as disaster-related (`target=1`) or not (`target=0`), while the test set has 3,263 tweets.
# 
# A preview of the data shows each tweet includes optional `keyword` and `location` fields, both of which have many missing values. There are 221 unique keywords and over 3,000 unique locations. The `text` and `target` columns are complete.

# %%
base = "nlp-getting-started/"
df_train = pd.read_csv(base + "train.csv")
df_test = pd.read_csv(base + "test.csv")

print('Training Set Shape = {}'.format(df_train.shape))
print('Test Set Shape = {}'.format(df_test.shape))

# %%
df_train.head()

# %%
df_test.head()

# %%
df_train.nunique()

# %%
display(df_train.isnull().sum())
display(df_test.isnull().sum())

# %%
print(f"Number of unique keywords: {df_train['keyword'].nunique()}")
print(f"Number of unique locations: {df_train['location'].nunique()}")

# %% [markdown]
# # Cleaning
# 
# To prepare the tweets for modeling, we applied several text cleaning steps. First, contractions are expanded (e.g., "can't" → "cannot") for consistency. URLs, usernames, and hashtags are removed or normalized, and text is converted to lowercase. We strip out digits, punctuation, and the HTML entity "amp". The text is then tokenized, stopwords are removed, and each word is lemmatized to its base form to reduce inflectional variation.
# 
# After cleaning, we check for duplicate tweets. Duplicates with the same label are dropped to avoid bias, and tweets with conflicting labels are removed entirely to ensure label consistency.

# %%
def clean_text(text):
    # Expand contractions (e.g., "I'm" → "I am")
    text = contractions.fix(text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Split CamelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Lowercase
    text = text.lower()

    # Remove usernames (e.g., @user)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags but keep the word (e.g., "#fire" → "fire")
    text = re.sub(r'#', '', text)

    # Remove $amp
    text = re.sub(r'\bamp\b', '', text)

    # Remove digits and punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords (optional)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize (optional but helps LSTM generalize)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]

    # Join tokens back into a string
    text = ' '.join(tokens)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_wordnet_pos(word):
    """Map POS tag to first character for WordNetLemmatizer"""
    from nltk import pos_tag
    tag = pos_tag([word])[0][1][0].lower()
    return {'n': wordnet.NOUN, 'v': wordnet.VERB, 'a': wordnet.ADJ, 'r': wordnet.ADV}.get(tag, wordnet.NOUN)

df_train['clean_text'] = df_train['text'].apply(lambda text: clean_text(text))
df_test['clean_text'] = df_test['text'].apply(lambda text: clean_text(text))

# %%
print("duplicated tweet & target: ", df_train[['clean_text', 'target']].duplicated().sum())
print("duplicated tweets: ", df_train[['clean_text']].duplicated().sum())

print(f"df size before dupe removal: {df_train.shape[0]}")

# Drop duplicates that have the same target
df_train = df_train.drop_duplicates(subset=['clean_text', 'target'], keep='first')

# Identify conflicting 'clean_text' entries with different 'target' values and remove
df_conflicts = df_train.groupby('clean_text').filter(lambda x: x['target'].nunique() > 1)
df_train = df_train[~df_train['clean_text'].isin(df_conflicts['clean_text'])]

print(f"df size after dupe removal: {df_train.shape[0]}")

# %% [markdown]
# # EDA
# 
# Exploratory Data Analysis (EDA) helps us understand the cleaned dataset and informs feature engineering and modeling.
# 
# 1.  **Word Clouds for Disaster and Non-Disaster Tweets:** (See Word Clouds plot) These visualize the most frequent words in each class, highlighting terms like "fire," "emergency," and "evacuate" for disaster tweets, and more generic or unrelated words for non-disaster tweets. This helps identify class-specific vocabulary, guiding the decision to use keywords as a feature and to filter out common words that appear in both classes.
# 2.  **Target Variable Distribution (Count Plot):** (See Count Plot) The count plot shows the number of disaster (target=1) and non-disaster (target=0) tweets. The classes are slightly imbalanced, with more non-disaster tweets. Recognizing this imbalance is important for model training, leading to the use of class weighting during RNN training.
# 3.  **Boxplot of Character Length by Target:** (See Boxplot) This plot compares tweet lengths across classes, showing that disaster tweets tend to have a slightly higher median character count than non-disaster tweets. This suggests that tweet length could be a useful feature for classification and supports using tokenization and padding which implicitly handle length.
# 4.  **Histograms of Character and Word Count by Target:** (See Histograms) Histograms display the distribution of tweet lengths (in characters and words) for each class. There is overlap, but disaster tweets are more likely to be longer. This insight supports the inclusion of length-based considerations in preprocessing and model architecture.
# 5.  **Top Keyword Frequencies (Bar Plot):** (See Keyword Frequency plot) The bar plot of the top 20 keywords shows that certain keywords, such as "fatalities," "damage," and "evacuation," are much more frequent and often associated with disaster tweets. This strongly supports the decision to include keywords as a feature by prepending them to the tweet text.
# 6.  **Stacked Bar Plot of Top Keywords by Disaster Proportion:** (See Stacked Bar Plot) This visualization shows how often each of the top 50 keywords appears in disaster versus non-disaster tweets. Some keywords are strong indicators of disasters, further supporting their use as predictive features.

# %%
# Create a figure and a set of subplots, one for each word cloud
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Word cloud for disaster tweets
disaster_text = ' '.join(df_train[df_train['target'] == 1]['clean_text'].astype(str).tolist())
wordcloud_disaster = WordCloud(width=800, height=400,
                                background_color='#FFEEEE',
                                stopwords=STOPWORDS,
                                min_font_size=10).generate(disaster_text)

# Display the disaster word cloud in the first subplot
ax1.imshow(wordcloud_disaster, interpolation='bilinear')
ax1.axis("off")
ax1.set_title('Word Cloud for Disaster Tweets')
ax1.set_facecolor('#FFEEEE')

# Word cloud for non-disaster tweets
nondisaster_text = ' '.join(df_train[df_train['target'] == 0]['clean_text'].astype(str).tolist())
wordcloud_nondisaster = WordCloud(width=800, height=400,
                                   background_color='#EEFFEE',
                                   stopwords=STOPWORDS,
                                   min_font_size=10).generate(nondisaster_text)

# Display the non-disaster word cloud in the second subplot
ax2.imshow(wordcloud_nondisaster, interpolation='bilinear')
ax2.axis("off")
ax2.set_title('Word Cloud for Non-Disaster Tweets')
ax2.set_facecolor('#EEFFEE')

plt.tight_layout()
plt.show()

# %%
# Create the count plot
sns.countplot(x='target', data=df_train)
plt.title('Distribution of Target Variable')
plt.xlabel('Target (0: Non-Disaster, 1: Disaster)')
plt.ylabel('Count')

plt.show()

# %%
df_train['char_length'] = df_train['text'].apply(lambda x: len(x))
df_train['word_length'] = df_train['text'].apply(lambda x: len(x.split()))

sns.boxplot(x='target', y='char_length', data=df_train)
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of Character Length
sns.histplot(data=df_train, x='char_length', hue='target', kde=True, ax=axes[0])
axes[0].set_title('Distribution of Character Count by Target')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].legend(title='Target', labels=['Non-Disaster (0)', 'Disaster (1)'])

# Histogram of Word Length
sns.histplot(data=df_train, x='word_length', hue='target', kde=True, ax=axes[1])
axes[1].set_title('Distribution of Word Count by Target')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].legend(title='Target', labels=['Non-Disaster (0)', 'Disaster (1)'])

plt.tight_layout()
plt.show()

# %%
# Replace NaN with 'no_keyword'
keywords = df_train['keyword'].fillna('no keyword').tolist()
keyword_counts = Counter(keywords)

# Get the most common keywords and their counts
top_keywords = keyword_counts.most_common(20)
keyword_labels, keyword_frequencies = zip(*top_keywords)

plt.figure(figsize=(12, 6))
plt.bar(keyword_labels, keyword_frequencies)
plt.xticks(rotation=90)
plt.title('Top 20 Keyword Frequencies')
plt.xlabel('Keywords')
plt.ylabel('Frequency')
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Get the most frequent keywords and filter dataset
top_keywords = df_train['keyword'].value_counts().head(50).index
filtered_df = df_train[df_train['keyword'].isin(top_keywords)]

# Compute disaster proportions and sort
keyword_counts = filtered_df.groupby(['keyword', 'target']).size().unstack(fill_value=0)
keyword_counts['disaster_proportion'] = keyword_counts[1] / (keyword_counts[0] + keyword_counts[1])
keyword_counts_sorted = keyword_counts.sort_values(by='disaster_proportion', ascending=False).drop(columns=['disaster_proportion'])

# Plot the bar chart
ax = keyword_counts_sorted.plot(kind='bar', stacked=True, figsize=(20, 6), color=['blue', 'red'])

plt.title("Top 50 Keywords Sorted by Disaster Tweet Proportion")
plt.xlabel("Keyword")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(["Non-Disaster (0)", "Disaster (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %% [markdown]
# # RNN Models

# %% [markdown]
# ## Preprocessing
# 
# For RNN preprocessing, several steps are applied to optimize input for deep learning models. First, the "keyword" field is cleaned, then, if present, the keyword is attached to the beginning of each tweet's cleaned text. This leverages the fact that certain keywords are strong predictors of disaster tweets, as we discovered during our EDA.
# 
# Next, common words that appear with nearly equal frequency in both disaster and non-disaster tweets are identified and removed, as they are unlikely to help the model distinguish between classes. This is done by calculating the proportion of each word's occurrence in both classes and filtering out those with proportions centered around 50% (0.3 to 0.7). Removing common words could reduce the generalizability of our models to unseen tweets, but upon testing the models with and without these words, performance was improved on the training validation set with these words removed. The Kaggle test score also improved.
# 
# The cleaned text is then tokenized using a Keras Tokenizer, limited to the top 10,000 words, and converted into sequences of integers. These sequences are padded to a uniform maximum length, ensuring consistent input shape for the RNN. The data is split into training and validation sets.

# %%
df_train['keyword'] = df_train['keyword'].str.replace('%20', '')
df_test['keyword'] = df_test['keyword'].str.replace('%20', '')

df_train["clean_text"] = df_train.apply(
    lambda row: f"{row['keyword'].replace(' ', '')} {row['clean_text']}" if pd.notna(row['keyword']) else row['clean_text'], axis=1
)

df_test['clean_text'] = df_test.apply(
    lambda row: f"{row['keyword'].replace(' ', '')} {row['clean_text']}" if pd.notna(row['keyword']) else row['clean_text'], axis=1
)

# %%
from collections import Counter

disaster_text = ' '.join(df_train[df_train['target'] == 1]['clean_text'].astype(str).tolist())
nondisaster_text = ' '.join(df_train[df_train['target'] == 0]['clean_text'].astype(str).tolist())

# Count occurrences of each word
d_counter = Counter(disaster_text.split())
nd_counter = Counter(nondisaster_text.split())

# Find common words between both texts
all_words = set(d_counter.keys()).intersection(nd_counter.keys())

# Calculate proportion for each common word and filter words
word_props = {word: d_counter[word] / (d_counter[word] + nd_counter[word]) for word in all_words}
filter_words = [word for word, prop in word_props.items() if 0.3 < prop < 0.7]

df_train['clean_text'] = df_train['clean_text'].apply(lambda text: ' '.join(word for word in text.split() if word not in filter_words))

print("Number of filtered words:", len(filter_words))
total_filters = [d_counter[word] + nd_counter[word] for word in filter_words]
print("Filtered word occurences:", sum(total_filters))
print("Total word occurrences:", sum(len(row['clean_text'].split()) for _, row in df_train.iterrows()))


# %%
vocab_size = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df_train['clean_text'])

train_sequences = tokenizer.texts_to_sequences(df_train['clean_text'])
test_sequences = tokenizer.texts_to_sequences(df_test['clean_text'])

max_length = max(len(seq) for sequences in [train_sequences, test_sequences] for seq in sequences)
train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

X_train, X_val, y_train, y_val = train_test_split(train_sequences, df_train['target'], test_size=0.2, random_state=42)
X_test = test_sequences

# %%
# Model hyperparameters
print(f"Tokenizer Vocabulary Size: {len(tokenizer.word_index) + 1}")
embedding_dim = 1028
rnn_units = 64
dense_units = 128
dense_dropout = 0.3

# Training hyperparameters
epochs = 50
learning_rate = 0.00005
patience = 3
batch_size = 32
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

# Dictionary to keep models for analysis
rnn_models = dict()

# Helper functions

# Insert RNN layers into sequential neural network
def build_rnn(rnn_layers, name=None, learning_rate=learning_rate):
    # Prepare input
    input = tf.keras.layers.Input(shape=(max_length,))
    embed = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(input)

    # Insert RNN layer(s)
    x = embed
    for rnn_layer in rnn_layers:
        x = rnn_layer(x)

    # Prepare Dense / Output layers
    dense = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    drop = tf.keras.layers.Dropout(dense_dropout)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(drop)

    model = tf.keras.models.Model(inputs=input, outputs=output, name=name)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    model.summary()
    return model

# Train the model
def train_rnn(model, X_train=X_train, y_train=y_train, epochs=epochs, batch_size=batch_size, patience=patience):
    early_stop = EarlyStopping(monitor='val_loss', mode='auto', patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stop]
    )
    return model

# Evaluate the model and visualize
def evaluate_model(model, X=X_val, y=y_val):
    y_pred = (model.predict(X, verbose=0)>0.5).astype(int)
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm).plot()


# %% [markdown]
# ## RNN Building and Training
# 
# We built and trained several Recurrent Neural Network (RNN) models to classify tweets, leveraging the sequential nature of text data. These models incorporate an **Embedding layer** to represent words as vectors and **Dense layers** for classification. The architectures fall into two categories: **sequential models (1–6)**, which process text step by step, and **parallel models (7–10)**, which combine multiple layers to extract different features simultaneously.
# 
# While training these models was computationally intensive, extensive experimentation was conducted to optimize performance. Each architecture was trained with multiple configurations, varying hyperparameters such as batch size, learning rate, layer sizes, and dropout rates. Additionally, different preprocessing choices, including the number of tokenized words and filtered vocabulary, were tested to assess their impact on classification accuracy. Given the sheer number of trial runs, including every iteration within this notebook would make it prohibitively large. The hyperparameters presented here reflect the most effective configurations identified through rigorous testing.
# 
# **Sequential Models**
# 
# 1. **Long Short-Term Memory**  
# Long Short-Term Memory (LSTM) networks are designed to handle sequential data by maintaining an internal memory state, allowing them to capture long-range dependencies while mitigating the vanishing gradient problem. Tweets often contain contextual cues spread across multiple words, making LSTMs effective in preserving meaningful relationships within text.
# 
# 2. **Gated Recurrent Unit**  
# Gated Recurrent Units (GRUs) are a variant of RNN with a simpler structure and fewer parameters. This reduces computational overhead while maintaining comparable performance, making them a more efficient alternative for processing sequential text.
# 
# 3. **Bidirectional Long Short-Term Memory**
# Bidirectional Long Short-Term Memory (Bi-LSTM) networks process text in both forward and backward directions, improving the model’s ability to understand word relationships by incorporating both future and past context. This bidirectional approach enhances text classification by capturing nuances often missed in standard LSTMs.
# 
# 4. **Bidirectional Gated Recurrent Unit**  
# Bidirectional Gated Recurrent Units (Bi-GRU) function similarly to the Bi-LSTM but uses GRU cells, offering efficiency benefits while maintaining the bidirectional context awareness. This model balances computational cost and sequential learning.
# 
# 5. **Convolutional Neural Network → Bidirectional Long Short-Term Memory**  
# Convolutional Neural Networks (CNNs) extract local features from text, before passing them to a Bi-LSTM. This combination allows the model to capture both local patterns and long-range dependencies, improving classification performance.
# 
# 6. **Convolutional Neural Network → Bidirectional Gated Recurrent Unit**  
# This variation replaces the Bi-LSTM with a Bi-GRU while maintaining the Convolutional Neural Network feature extraction step. It retains bidirectional sequential processing but with a lighter architecture, potentially reducing training time.
# 
# **Parallel Models**
# 
# 7. **Long Short-Term Memory + Gated Recurrent Unit**  
# This model integrates both LSTM and GRU layers in parallel, potentially allowing the network to leverage the strengths of each. Combining these architectures could potentially leverage the strengths of both in terms of identifing sequential features.
# 
# 8. **Bidirectional Long Short-Term Memory + Bidirectional Gated Recurrent Unit**  
# Bidirectional versions of Long Short-Term Memory and Gated Recurrent Unit networks are combined in a single parallel architecture to enhance sequential feature extraction, leveraging both bidirectional memory structures for improved text representation.
# 
# 9. **Convolutional Neural Network + Long Short-Term Memory + Gated Recurrent Unit**  
# A Convolutional Neural Network extracts local patterns, while both Long Short-Term Memory and Gated Recurrent Unit layers extract sequential features. This approach aims to combine local feature detection with sequential memory retention, providing a multi-layered feature extraction mechanism.
# 
# 10. **Convolutional Neural Network + Bidirectional Long Short-Term Memory + Bidirectional Gated Recurrent Unit**  
# The most complex model tested, integrating a Convolutional Neural Network with both Bidirectional Long Short-Term Memory and Bidirectional Gated Recurrent Unit layers. This setup merges local feature extraction with comprehensive bidirectional sequence processing, allowing the model to capture a wide range of patterns in text classification.

# %% [markdown]
# ### Model 1: LSTM

# %%
rnn_layers = [
    LSTM(rnn_units)
]

model = build_rnn(rnn_layers, 'LSTM')
model = train_rnn(model)
#evaluate_model(model, X_val, y_val)
rnn_models['LSTM'] = model

# %% [markdown]
# ### Model 2: GRU

# %%
rnn_layers = [
    tf.keras.layers.GRU(rnn_units)
]

model = build_rnn(rnn_layers, name='GRU')
model = train_rnn(model)
#evaluate_model(model, X_val, y_val)
rnn_models['GRU'] = model

# %% [markdown]
# ### Model 3: Bidirectional LSTM

# %%
rnn_layers = [
    tf.keras.layers.Bidirectional(LSTM(rnn_units))
]

model = build_rnn(rnn_layers, name='BD-LSTM')
model = train_rnn(model)
#evaluate_model(model, X_val, y_val)
rnn_models['BiLSTM'] = model

# %% [markdown]
# ### Model 4: Bidirectional GRU

# %%
rnn_layers = [
    tf.keras.layers.Bidirectional(GRU(rnn_units))
]

model = build_rnn(rnn_layers, name='BD-GRU')
model = train_rnn(model)
#evaluate_model(model, X_val, y_val)
rnn_models['BiGRU'] = model

# %% [markdown]
# ### Model 5: CNN -> Bidirectional LSTM

# %%
rnn_layers = [
    tf.keras.layers.Conv1D(filters=rnn_units, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Bidirectional(LSTM(rnn_units))
]

model = build_rnn(rnn_layers, 'Conv-BD-LSTM')
model = train_rnn(model)
#evaluate_model(model, X_val, y_val)
rnn_models['CNN > BiLSTM'] = model

# %% [markdown]
# ### Model 6: CNN -> Bidirectional GRU

# %%
from tensorflow.keras.regularizers import l2

rnn_layers = [
    tf.keras.layers.Conv1D(filters=rnn_units, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Bidirectional(GRU(rnn_units))
]

model = build_rnn(rnn_layers, name="Conv-BD-GRU")
model = train_rnn(model, epochs=50)
#evaluate_model(model, X_val, y_val)
rnn_models['CNN > BiGRU'] = model

# %% [markdown]
# ### Model 7: LSTM + GRU

# %%
# Input and embedding layers
inputs = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

# RNN Branches and Concatenation
gru = GRU(rnn_units, recurrent_dropout=0.2)(embedding)
lstm = LSTM(rnn_units, recurrent_dropout=0.2)(embedding)
merged = Concatenate()([gru, lstm])

# Dense Layers
dense = Dense(dense_units, activation='relu')(merged)
dropout = Dropout(dense_dropout)(dense)
output = Dense(1, activation='sigmoid')(dropout)

# Model definition
model = Model(inputs, output)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

train_rnn(model, batch_size=128)
rnn_models['LSTM + GRU'] = model

# %% [markdown]
# ### Model 8: BiLSTM + BiGRU

# %%
# Input and embedding layers
inputs = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

# RNN Branches and Concatenation
gru = Bidirectional(GRU(rnn_units, recurrent_dropout=0.2))(embedding)
lstm = Bidirectional(LSTM(rnn_units, recurrent_dropout=0.2))(embedding)
merged = Concatenate()([gru, lstm])

# Dense Layers
dense = Dense(dense_units, activation='relu')(merged)
dropout = Dropout(dense_dropout)(dense)
output = Dense(1, activation='sigmoid')(dropout)

# Model definition
model = Model(inputs, output)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

train_rnn(model)
rnn_models['BiLSTM + BiGRU'] = model

# %% [markdown]
# ### Model 9: CNN + LSTM + GRU

# %%
# Input and embedding layers
inputs = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

# Conv1D and RNN Branches and Concatenation
conv1d = Conv1D(filters=rnn_units, kernel_size=3, activation='relu')(embedding)
conv1d_pooled = GlobalMaxPooling1D()(conv1d)
gru = GRU(rnn_units, recurrent_dropout=0.2)(embedding)
lstm = LSTM(rnn_units, recurrent_dropout=0.2)(embedding)
merged = Concatenate()([conv1d_pooled, gru, lstm])

# Dense Layers
dense = Dense(dense_units, activation='relu')(merged)
dropout = Dropout(dense_dropout)(dense)
output = Dense(1, activation='sigmoid')(dropout)

# Model definition
model = Model(inputs, output)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

train_rnn(model)
rnn_models['CNN + LSTM + GRU'] = model

# %% [markdown]
# ### Model 10: CNN + BiLSTM + BiGRU

# %%
# Input and embedding layers
inputs = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

# Conv1D and RNN Branches and Concatenation
conv1d = Conv1D(filters=rnn_units, kernel_size=3, activation='relu')(embedding)
conv1d_pooled = GlobalMaxPooling1D()(conv1d)  # Reducing feature maps to a fixed-length vector
gru = Bidirectional(GRU(rnn_units, recurrent_dropout=0.2))(embedding)
lstm = Bidirectional(LSTM(rnn_units, recurrent_dropout=0.2))(embedding)
merged = Concatenate()([conv1d_pooled, gru, lstm])

# Dense Layers
dense = Dense(dense_units, activation='relu')(merged)
dropout = Dropout(dense_dropout)(dense)
output = Dense(1, activation='sigmoid')(dropout)  # Binary classification

# Model definition
model = Model(inputs, output)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

train_rnn(model)
rnn_models['CNN + BiLSTM + BiGRU'] = model

# %% [markdown]
# ## Analysis of RNN Model Performance
# 
# To assess the performance of the various RNN architectures, we evaluated several key metrics: Accuracy, F1-score, Precision, and Recall. A bar plot (refer to the "RNN Model Performance Comparison Across Metrics" plot) visualizes these scores across the different models. From this visualization, we can observe variations in performance. For example, the BiLSTM model exhibits the highest F1-score, indicating a good balance between precision and recall. In contrast, the CNN > BiGRU model shows a high precision but lower recall, suggesting it is good at identifying true positives but misses more actual positive cases. The ROC curve plot further illustrates the trade-off between true positive rate and false positive rate for each model. The area under the curve (AUC) provides a single measure of overall performance. The BiLSTM model again demonstrates a high AUC, reinforcing its strong performance. Examining the confusion matrices for the top three models by F1-score (refer to the "Confusion Matrix for top 3 performing models" plots) provides a more detailed breakdown of their predictions, showing the number of true positives, true negatives, false positives, and false negatives. The BiLSTM confusion matrix, for instance, shows a good number of true positives and true negatives relative to false positives and negatives.
# 
# Based on the F1-score, the BiLSTM+BiGRU model is identified as the best performing model with an F1-score of 0.781163. This architecture consists of a Bidirectional LSTM layer and Bidirectional GRU layer running in parallel after the embedding layer. The Bidirectional architectural is likely a key contributor to its success. By processing the text sequence in both forward and backward directions, the model can capture contextual information from both the beginning and end of a tweet, which is particularly beneficial in understanding the nuances of natural language. It leverages both GRU and LSTM and perhaps benefits from the strengths of both. Another strength lies in its ability to retain long-term dependencies and its bidirectional nature allows for a more complete understanding of the tweet's meaning. Other competitive models include the BiGRU and BiLSTM, which suggests that the bidirectional architecture is identifying features that the other models are not.
# 
# On the Kaggle competition leaderboard (see kaggle_score.png), the best performing model achieved a respectable public score of 0.78026. This indicates that the chosen model and preprocessing steps were effective in generalizing to unseen data, but improvements could be made.

# %%
# Calculate metrics and prepare results for analysis
rnn_results = []
for model_name, model in rnn_models.items():
    #print(f"Model: {model_name}")
    yp = (model.predict(X_val, verbose=0)>0.5).astype(int)
    #print(classification_report(y_val, y_val_pred))
    #print()
    acc = accuracy_score(y_val, yp)
    f1 = f1_score(y_val, yp)
    auc = roc_auc_score(y_val.values, yp)
    prec = precision_score(y_val.values, yp)
    rec = recall_score(y_val.values, yp)
    cm = confusion_matrix(y_val, yp)
    #print(f"{model_name:6}: acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}, prec={prec:.3f}, rec={rec:.3f}")
    result = {'name': model_name, 'acc': acc, 'f1': f1, 'auc': auc, 'prec': prec, 'rec': rec, 'cm': cm, 'model': model}
    rnn_results.append(result)

rnn_results_df = pd.DataFrame(rnn_results)
rnn_results_df.drop(columns='model').head(10)

# %%
# 1. Bar plot of metrics for rnn models
results_melted_rnn = rnn_results_df.melt(id_vars='name',
                                         value_vars=['acc', 'f1', 'prec', 'rec'],
                                         var_name='Metric',
                                         value_name='Score')

plt.figure(figsize=(15, 6))
sns.barplot(x='name', y='Score', hue='Metric', data=results_melted_rnn)

plt.title('RNN Model Performance Comparison Across Metrics')
plt.xlabel('RNN Model')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 2. ROC Curve
plt.figure(figsize=(10, 8))
for index, row in rnn_results_df.iterrows():
    model = row['model']
    model_name = row['name']

    y_pred_prob = model.predict(X_val, verbose=0)

    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
    auc = roc_auc_score(y_val, y_pred_prob)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - RNN Models')
plt.legend()
plt.show()

# 3. Confusion Matrix for top 3 performing models
top_models = rnn_results_df.sort_values(by='f1', ascending=False).head(3)
num_models = len(top_models)
fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
fig.subplots_adjust(wspace=0.5)

for index, (ax, row) in enumerate(zip(axes, top_models.iterrows())):
    model_name = row[1]['name']
    cm = row[1]['cm']
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'Confusion Matrix for {model_name}')

plt.tight_layout()
plt.show()


# %%
# Find the model with the highest F1-score.
best_model = rnn_results_df.loc[rnn_results_df['f1'].idxmax()]
print(f"The best model is {best_model['name']}")
evaluate_model(best_model['model'])

# Generate predictions on test set
df_test['target'] = (best_model['model'].predict(test_sequences, verbose=0)>0.5).astype(int)
df_test[['id', 'target']].to_csv('best_predictions.csv', index=False)


# %%
# Kaggle score result
from IPython.display import Image
Image(filename="kaggle_score.png")

# %% [markdown]
# # Other models
# 
# To compare the performance of the developed RNN models, several traditional supervised learning models were also trained and evaluated on the dataset. These models provide a benchmark to understand how deep learning approaches fare against more classical techniques for this classification task.

# %% [markdown]
# ## Preprocessing
# 
# For traditional supervised learning models, cleaned tweet text was converted into numerical feature vectors using TF-IDF Vectorization. The `TfidfVectorizer` was configured to limit features to the most frequent and ignore terms appearing in fewer than 5 tweets. The resulting feature matrix and target variable were split into training and validation sets (80/20 split). Class weights were computed to address the dataset's class imbalance during model training.

# %%
vectorizer = TfidfVectorizer(norm='l2', ngram_range=(1,3), max_features=vocab_size, sublinear_tf=True, min_df=5)
vectorizer = vectorizer.fit(df_train['clean_text'])

vocabulary_shape = vectorizer.vocabulary_
print(f"Vocabulary shape (number of features): {len(vocabulary_shape)}")

X = vectorizer.transform(df_train['clean_text'])
print(df_train.shape[0], X.shape[0])

y = df_train['target']

# %%
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X, y, train_size=0.8, random_state=42)

class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train_s), y=y_train_s)))

sup_models = dict()

# %% [markdown]
# ## Building and training
# 
# The supervised models used were constructed and trained subsequent to hyperparameter optimization conducted via GridSearchCV. This systematic approach explores a predefined range of hyperparameter values for each model, utilizing cross-validation to identify the configuration that maximizes performance based on a specified scoring metric, in this case, the F1-score. The F1-score was selected as the primary evaluation metric during grid search mainly because the Kaggle contest scoring is based on F1 but also because of its balance between precision and recall.
# 
# Models explored:
# 
# 1.  **Random Forest Classifier (RF):** An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. It is robust to overfitting and can handle a large number of features.
# 
# 2.  **Support Vector Classifier (SVC):** A supervised learning model used for classification and regression analysis. It finds the optimal hyperplane that best separates data points of different classes in a high-dimensional space.
# 
# 3.  **XGBoost:** An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.
# 
# 4.  **Logistic Regression:** A statistical model that uses a logistic function to model a binary dependent variable. It's a linear model that estimates the probability of a binary outcome.

# %% [markdown]
# ### Random Forest Classifier (RF)

# %%
grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight=class_weights, random_state=42),
    param_grid={
        'n_estimators': range(50, 201, 50),
        'max_depth': range(5, 21, 5),
        'min_samples_split': range(2, 6),
        'min_samples_leaf': range(1, 4)
    },
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_rf.fit(X_train_s, y_train_s)
model_rf = grid_rf.best_estimator_

print(f'Best Parameters: {grid_rf.best_params_}')
print(f'Best score: {grid_rf.best_score_}')

sup_models['Random Forest'] = model_rf

# %% [markdown]
# ### Support Vector Classifier (SVC)

# %%
grid_svc = GridSearchCV(
    SVC(class_weight=class_weights, random_state=42),
    param_grid={
        'C': [0.5, 1, 2, 5, 10],
        'gamma': [0.1, 0.25, 0.5, 0.75, 1],
        'kernel': ['linear', 'rbf']
    },
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_svc.fit(X_train_s, y_train_s)
model_svc = grid_svc.best_estimator_

print(f'Best Parameters: {grid_svc.best_params_}')
print(f'Best score: {grid_svc.best_score_}')

sup_models['SVC'] = model_svc

# %% [markdown]
# ### XGBoost

# %%
grid_xgb = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid={
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]  # Fraction of features used
    },
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_xgb.fit(X_train_s, y_train_s)
model_xgb = grid_xgb.best_estimator_

print(f'Best Parameters: {grid_xgb.best_params_}')
print(f'Best Score: {grid_xgb.best_score_}')

sup_models['XGBoost'] = model_xgb

# %% [markdown]
# ### Logistic Regression

# %%
grid_lr = GridSearchCV(
    LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    param_grid={
        'C': [0.1, 0.5, 1, 5, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_lr.fit(X_train_s, y_train_s)
model_lr = grid_lr.best_estimator_

print(f'Best Parameters: {grid_lr.best_params_}')
print(f'Best Score: {grid_lr.best_score_}')

sup_models['Logistic Regression'] = model_lr

# %% [markdown]
# ## Analysis of Other Model Performance
# 
# To provide context for the RNN models' performance, we also evaluated several traditional supervised learning models: Random Forest, SVC, XGBoost, and Logistic Regression. We will compare their performance against the general performance observed across the RNN models.
# 
# First, considering the Random Forest model, its performance as shown in the "Supervised Model Performance Comparison Across Metrics" bar plot indicates lower scores across all metrics (Accuracy, F1, Precision, Recall) compared to most of the RNN models. The confusion matrix for Random Forest highlights a significant number of false negatives, indicating that it struggles to correctly identify disaster tweets. This suggests that while Random Forest is a powerful algorithm, the nature of the tweet data, with its sequential and contextual dependencies, might not be optimally captured by this model compared to RNNs.
# 
# Next, the Support Vector Classifier (SVC) demonstrates noticeably better performance than Random Forest, with higher scores across all metrics. Observing the bar plot, SVC's F1-score and Accuracy are closer to the lower end of the RNN models' performance range. Its confusion matrix shows a more balanced distribution of errors compared to Random Forest, but it still has a considerable number of false negatives and false positives when compared to the better-performing RNNs. SVC, being a non-sequential model, might not fully leverage the ordering of words in a tweet, which is where RNNs excel.
# 
# The XGBoost model, a gradient boosting algorithm, also shows improved performance compared to Random Forest. However, its F1-score and Recall are still generally lower than the majority of the RNN models, as seen in the bar plot. The XGBoost confusion matrix reveals a relatively high number of false negatives, similar to Random Forest, suggesting difficulty in capturing all positive instances. While XGBoost is known for its strong performance on various tabular datasets, it may not be as well-suited for sequential text data compared to architectures specifically designed for it, like RNNs.
# 
# Finally, Logistic Regression, a linear model, shows performance comparable to SVC and XGBoost, with an F1-score and Accuracy within the mid-range of the supervised models but generally lower than the RNNs. Its confusion matrix shows a similar pattern of errors to SVC. As a linear model, Logistic Regression's ability to capture complex, non-linear relationships and dependencies in the text data might be limited compared to deep learning models.
# 
# Comparing the supervised models to the RNN models generally, the RNN models exhibit superior performance across most evaluation metrics, particularly F1-score and AUC. This suggests that deep learning architectures, specifically designed to handle sequential data, are better equipped to capture the intricate patterns and dependencies within tweet text for this classification task.
# 
# Among the traditional supervised models, the SVC model appears to be the best performing based on the F1-score (0.7455). However, comparing this to the best performing RNN model, the BiLSTM+BiGRU, the latter still holds an advantage. The top RNN model's ability to process sequences bidirectionally and capture long-range dependencies likely contributes to this superior performance on the tweet classification task.

# %%
# Calculate metrics and prepare results for analysis

sup_results = []
for model_name, model in sup_models.items():
    yp = (model.predict(X_val_s)>0.5).astype(int)
    acc = accuracy_score(y_val_s, yp)
    f1 = f1_score(y_val_s, yp)
    auc = roc_auc_score(y_val_s.values, yp)
    prec = precision_score(y_val_s.values, yp)
    rec = recall_score(y_val_s.values, yp)
    cm = confusion_matrix(y_val_s, yp)
    result = {'name': model_name, 'acc': acc, 'f1': f1, 'auc': auc, 'prec': prec, 'rec': rec, 'cm': cm, 'model': model}
    sup_results.append(result)

sup_results_df = pd.DataFrame(sup_results)

sup_results_df.drop(columns='model').head(10)

# %%
for item in rnn_results:
    item['type'] = 'RNN'
for item in sup_results:
    item['type'] = 'Other'

results_df = pd.DataFrame(rnn_results + sup_results)

# %%
# 1. Bar plot of metrics for all models

all_results_melted = results_df.melt(id_vars='name',
                                     value_vars=['acc', 'f1', 'prec', 'rec'],
                                     var_name='Metric',
                                     value_name='Score')

plt.figure(figsize=(18, 8))
sns.barplot(x='name', y='Score', hue='Metric', data=all_results_melted)

plt.title('Model Performance Comparison Across Metrics')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric', loc='center right', bbox_to_anchor=(1, 0.5))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. ROC Curve Plot
plt.figure(figsize=(15, 8))

# Plot ROC curves for RNN models
for index, row in rnn_results_df.iterrows():
    model = row['model']
    model_name = row['name']
    # Assuming X_val is the validation data for RNNs
    y_pred_proba = model.predict(X_val, verbose=0)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f'RNN: {model_name} (AUC = {auc:.2f})')

# Plot ROC curves for Supervised models
for index, row in sup_results_df.iterrows():
    model = row['model']
    model_name = row['name']
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val_s)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val_s, y_pred_proba)
        auc = roc_auc_score(y_val_s, y_pred_proba)
        plt.plot(fpr, tpr, label=f'Sup: {model_name} (AUC = {auc:.2f})')
    elif hasattr(model, 'decision_function'):
         y_pred_proba = model.decision_function(X_val_s)
         fpr, tpr, thresholds = roc_curve(y_val_s, y_pred_proba)
         auc = roc_auc_score(y_val_s, y_pred_proba)
         plt.plot(fpr, tpr, label=f'Sup: {model_name} (AUC = {auc:.2f})')


plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Find the model with the highest F1-score.
best_model = results_df.loc[results_df['f1'].idxmax()]

print(f"The best model is {best_model['name']}")

if best_model['name'] in rnn_models:
    y_pred = (best_model['model'].predict(X_val)>0.5).astype(int)
    print(classification_report(y_val, y_pred))
else:
    y_pred = (best_model['model'].predict(X_val_s)>0.5).astype(int)
    print(classification_report(y_val_s, y_pred))

# %% [markdown]
# # Conclusion
# 
# In this project, we focused on classifying tweets as either disaster-related or not using the Kaggle "Natural Language Processing with Disaster Tweets" dataset. Following initial data preparation, we extensively explored various deep learning Recurrent Neural Network (RNN) architectures and compared their performance to traditional supervised learning models.
# 
# Our analysis of the RNN models revealed a range of effectiveness, with the BiLSTM architecture demonstrating superior performance based on the F1-score and AUC metrics. This success is likely attributable to its bidirectional nature, which allows the model to capture contextual information from both preceding and succeeding words in a tweet, crucial for understanding nuances in natural language.
# 
# Comparing the RNN models to traditional supervised learning methods like Random Forest, SVC, XGBoost, and Logistic Regression highlighted the advantage of deep learning for this sequential text classification task. The RNN models, as a group, generally outperformed the traditional models, indicating their better ability to learn complex patterns and dependencies within the tweet data. While the SVC model was the best among the traditional methods, the leading RNN model, BiLSTM, still achieved a higher F1-score.
# 
# Despite achieving a respectable score on the Kaggle leaderboard, indicating good generalization to unseen data, there are limitations and areas for improvement. Our preprocessing, while effective, could be further refined, for instance, by exploring more advanced techniques for handling emojis or slang. In addition, while removing words that are common to both classes improved performance, it does result in a loss of information that could impact performance on unseen data.
# 
# Future work could involve experimenting with more sophisticated deep learning architectures, such as Transformer networks, which have shown state-of-the-art results in various NLP tasks. Additionally, incorporating external data sources or using pre-trained language models like BERT could potentially boost performance. Further hyperparameter tuning and cross-validation on a larger scale could also lead to some gains although extensive hyperparameter tuning using these architectures resulted in only marginal gains. Addressing the class imbalance with more advanced techniques than class weighting, might also yield improvements.


