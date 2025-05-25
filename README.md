# NLP-Disaster-Tweets-Kaggle-Mini-Project
Week 4 Intro to Deep Learning Project

# Week 4 Assignment: NLP Disaster Tweets Kaggle Mini-Project

The goal of this project is to classify tweets as either disaster-related or non-disaster-related. This will be achieved using deep learning Recurrent Neural Networks (RNNs) and Natural Language Processing (NLP) techniques. We will be working with the ["Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started/data) dataset from Kaggle.  The project involves converting tweets into a format understandable by our model, training a series of RNNs to identify the best architecture for this classification task, and comparing its performance against several supervised learning models. Model performance will be evaluated using metrics such as accuracy and F1-score, the latter being the primary metric as it is the standard for judging performance in the Kaggle competition.


# Data Inspection

We began by loading the Kaggle "Natural Language Processing with Disaster Tweets" dataset. The training set contains 7,613 tweets, each labeled as disaster-related (`target=1`) or not (`target=0`), while the test set has 3,263 tweets.

A preview of the data shows each tweet includes optional `keyword` and `location` fields, both of which have many missing values. There are 221 unique keywords and over 3,000 unique locations. The `text` and `target` columns are complete.

# Cleaning

To prepare the tweets for modeling, we applied several text cleaning steps. First, contractions are expanded (e.g., "can't" → "cannot") for consistency. URLs, usernames, and hashtags are removed or normalized, and text is converted to lowercase. We strip out digits, punctuation, and the HTML entity "amp". The text is then tokenized, stopwords are removed, and each word is lemmatized to its base form to reduce inflectional variation.

After cleaning, we check for duplicate tweets. Duplicates with the same label are dropped to avoid bias, and tweets with conflicting labels are removed entirely to ensure label consistency.

# EDA

Exploratory Data Analysis (EDA) helps us understand the cleaned dataset and informs feature engineering and modeling.

1.  **Word Clouds for Disaster and Non-Disaster Tweets:** (See Word Clouds plot) These visualize the most frequent words in each class, highlighting terms like "fire," "emergency," and "evacuate" for disaster tweets, and more generic or unrelated words for non-disaster tweets. This helps identify class-specific vocabulary, guiding the decision to use keywords as a feature and to filter out common words that appear in both classes.
2.  **Target Variable Distribution (Count Plot):** (See Count Plot) The count plot shows the number of disaster (target=1) and non-disaster (target=0) tweets. The classes are slightly imbalanced, with more non-disaster tweets. Recognizing this imbalance is important for model training, leading to the use of class weighting during RNN training.
3.  **Boxplot of Character Length by Target:** (See Boxplot) This plot compares tweet lengths across classes, showing that disaster tweets tend to have a slightly higher median character count than non-disaster tweets. This suggests that tweet length could be a useful feature for classification and supports using tokenization and padding which implicitly handle length.
4.  **Histograms of Character and Word Count by Target:** (See Histograms) Histograms display the distribution of tweet lengths (in characters and words) for each class. There is overlap, but disaster tweets are more likely to be longer. This insight supports the inclusion of length-based considerations in preprocessing and model architecture.
5.  **Top Keyword Frequencies (Bar Plot):** (See Keyword Frequency plot) The bar plot of the top 20 keywords shows that certain keywords, such as "fatalities," "damage," and "evacuation," are much more frequent and often associated with disaster tweets. This strongly supports the decision to include keywords as a feature by prepending them to the tweet text.
6.  **Stacked Bar Plot of Top Keywords by Disaster Proportion:** (See Stacked Bar Plot) This visualization shows how often each of the top 50 keywords appears in disaster versus non-disaster tweets. Some keywords are strong indicators of disasters, further supporting their use as predictive features.
7.  **Proportion of Tweets with URLs by Target:** (See URL Proportion plot) A bar plot compares the proportion of tweets containing URLs in each class. Disaster tweets are slightly more likely to include URLs, indicating that URL presence could potentially be used as a binary feature (although this was not explicitly added as a separate feature in the final models, the cleaning process handles URLs).

![image](https://github.com/user-attachments/assets/3a5ff147-e967-4b2a-a8f9-2d9b2a86ad20)
![image](https://github.com/user-attachments/assets/2c0a8e49-7351-47d7-ae61-5841b7963262)
![image](https://github.com/user-attachments/assets/696f9a98-ef74-4011-add5-7527b218279e)
![image](https://github.com/user-attachments/assets/90bdddd3-646d-47b1-a728-beeaca6fb34c)
![image](https://github.com/user-attachments/assets/5d1b141d-2b04-459c-a075-9ed28bded160)
![image](https://github.com/user-attachments/assets/6f7bb25f-a276-473a-8a94-261b6dd5649d)

# RNN Models

## Preprocessing

For RNN preprocessing, several steps are applied to optimize input for deep learning models. First, the "keyword" field is cleaned, then, if present, the keyword is attached to the beginning of each tweet's cleaned text. This leverages the fact that certain keywords are strong predictors of disaster tweets, as we discovered during our EDA.

Next, common words that appear with nearly equal frequency in both disaster and non-disaster tweets are identified and removed, as they are unlikely to help the model distinguish between classes. This is done by calculating the proportion of each word's occurrence in both classes and filtering out those with proportions centered around 50% (0.3 to 0.7). Removing common words could reduce the generalizability of our models to unseen tweets, but upon testing the models with and without these words, performance was improved on the training validation set with these words removed. The Kaggle test score also improved.

The cleaned text is then tokenized using a Keras Tokenizer, limited to the top 10,000 words, and converted into sequences of integers. These sequences are padded to a uniform maximum length, ensuring consistent input shape for the RNN. The data is split into training and validation sets.

## RNN Building and Training

We built and trained several Recurrent Neural Network (RNN) models to classify tweets, leveraging the sequential nature of text data. These models incorporate an **Embedding layer** to represent words as vectors and **Dense layers** for classification. The architectures fall into two categories: **sequential models (1–6)**, which process text step by step, and **parallel models (7–10)**, which combine multiple layers to extract different features simultaneously.

While training these models was computationally intensive, extensive experimentation was conducted to optimize performance. Each architecture was trained with multiple configurations, varying hyperparameters such as batch size, learning rate, layer sizes, and dropout rates. Additionally, different preprocessing choices, including the number of tokenized words and filtered vocabulary, were tested to assess their impact on classification accuracy. Given the sheer number of trial runs, including every iteration within this notebook would make it prohibitively large. The hyperparameters presented here reflect the most effective configurations identified through rigorous testing.

**Sequential Models**

1. **Long Short-Term Memory**  
Long Short-Term Memory (LSTM) networks are designed to handle sequential data by maintaining an internal memory state, allowing them to capture long-range dependencies while mitigating the vanishing gradient problem. Tweets often contain contextual cues spread across multiple words, making LSTMs effective in preserving meaningful relationships within text.

2. **Gated Recurrent Unit**  
Gated Recurrent Units (GRUs) are a variant of RNN with a simpler structure and fewer parameters. This reduces computational overhead while maintaining comparable performance, making them a more efficient alternative for processing sequential text.

3. **Bidirectional Long Short-Term Memory**
Bidirectional Long Short-Term Memory (Bi-LSTM) networks process text in both forward and backward directions, improving the model’s ability to understand word relationships by incorporating both future and past context. This bidirectional approach enhances text classification by capturing nuances often missed in standard LSTMs.

4. **Bidirectional Gated Recurrent Unit**  
Bidirectional Gated Recurrent Units (Bi-GRU) function similarly to the Bi-LSTM but uses GRU cells, offering efficiency benefits while maintaining the bidirectional context awareness. This model balances computational cost and sequential learning.

5. **Convolutional Neural Network → Bidirectional Long Short-Term Memory**  
Convolutional Neural Networks (CNNs) extract local features from text, before passing them to a Bi-LSTM. This combination allows the model to capture both local patterns and long-range dependencies, improving classification performance.

6. **Convolutional Neural Network → Bidirectional Gated Recurrent Unit**  
This variation replaces the Bi-LSTM with a Bi-GRU while maintaining the Convolutional Neural Network feature extraction step. It retains bidirectional sequential processing but with a lighter architecture, potentially reducing training time.

**Parallel Models**

7. **Long Short-Term Memory + Gated Recurrent Unit**  
This model integrates both LSTM and GRU layers in parallel, potentially allowing the network to leverage the strengths of each. Combining these architectures could potentially leverage the strengths of both in terms of identifing sequential features.

8. **Bidirectional Long Short-Term Memory + Bidirectional Gated Recurrent Unit**  
Bidirectional versions of Long Short-Term Memory and Gated Recurrent Unit networks are combined in a single parallel architecture to enhance sequential feature extraction, leveraging both bidirectional memory structures for improved text representation.

9. **Convolutional Neural Network + Long Short-Term Memory + Gated Recurrent Unit**  
A Convolutional Neural Network extracts local patterns, while both Long Short-Term Memory and Gated Recurrent Unit layers extract sequential features. This approach aims to combine local feature detection with sequential memory retention, providing a multi-layered feature extraction mechanism.

10. **Convolutional Neural Network + Bidirectional Long Short-Term Memory + Bidirectional Gated Recurrent Unit**  
The most complex model tested, integrating a Convolutional Neural Network with both Bidirectional Long Short-Term Memory and Bidirectional Gated Recurrent Unit layers. This setup merges local feature extraction with comprehensive bidirectional sequence processing, allowing the model to capture a wide range of patterns in text classification.

## Analysis of RNN Model Performance

To assess the performance of the various RNN architectures, we evaluated several key metrics: Accuracy, F1-score, Precision, and Recall. A bar plot (refer to the "RNN Model Performance Comparison Across Metrics" plot) visualizes these scores across the different models. From this visualization, we can observe variations in performance. For example, the BiLSTM model exhibits the highest F1-score, indicating a good balance between precision and recall. In contrast, the CNN > BiGRU model shows a high precision but lower recall, suggesting it is good at identifying true positives but misses more actual positive cases. The ROC curve plot further illustrates the trade-off between true positive rate and false positive rate for each model. The area under the curve (AUC) provides a single measure of overall performance. The BiLSTM model again demonstrates a high AUC, reinforcing its strong performance. Examining the confusion matrices for the top three models by F1-score (refer to the "Confusion Matrix for top 3 performing models" plots) provides a more detailed breakdown of their predictions, showing the number of true positives, true negatives, false positives, and false negatives. The BiLSTM confusion matrix, for instance, shows a good number of true positives and true negatives relative to false positives and negatives.

Based on the F1-score, the BiLSTM model is identified as the best performing model with an F1-score of 0.7715. This architecture consists of a Bidirectional LSTM layer after the embedding layer. The Bidirectional LSTM layer is likely a key contributor to its success. By processing the text sequence in both forward and backward directions, the model can capture contextual information from both the beginning and end of a tweet, which is particularly beneficial in understanding the nuances of natural language. Another strength lies in its ability to retain long-term dependencies and its bidirectional nature allows for a more complete understanding of the tweet's meaning. Other competitive models include the BiGRU and CNN + LSTM + GRU, which also demonstrate respectable F1-scores and AUC values, suggesting that bidirectional layers and the inclusion of CNN layers can be beneficial.

On the Kaggle competition leaderboard (see kaggle_score.png), the best performing model achieved a respectable public score of 0.78026. This indicates that the chosen model and preprocessing steps were effective in generalizing to unseen data, but improvements could be made.

![image](https://github.com/user-attachments/assets/62717a17-98a2-42a9-a691-2a943664d2e6)

![image](https://github.com/user-attachments/assets/264af1ea-9552-4a29-9b3a-0cb92dff8b44)

![image](https://github.com/user-attachments/assets/207b326e-e5c9-44b4-b803-d6ce773ad537)

![image](https://github.com/user-attachments/assets/a85206f8-f8df-4e28-ba1c-8b66c70cd96d)

![image](https://github.com/user-attachments/assets/06d35d46-9eee-4c32-907c-72718e6fb2ed)

# Other models

To compare the performance of the developed RNN models, several traditional supervised learning models were also trained and evaluated on the dataset. These models provide a benchmark to understand how deep learning approaches fare against more classical techniques for this classification task.

## Preprocessing

For traditional supervised learning models, cleaned tweet text was converted into numerical feature vectors using TF-IDF Vectorization. The `TfidfVectorizer` was configured to limit features to the most frequent and ignore terms appearing in fewer than 5 tweets. The resulting feature matrix and target variable were split into training and validation sets (80/20 split). Class weights were computed to address the dataset's class imbalance during model training.

## Building and training

The supervised models used were constructed and trained subsequent to hyperparameter optimization conducted via GridSearchCV. This systematic approach explores a predefined range of hyperparameter values for each model, utilizing cross-validation to identify the configuration that maximizes performance based on a specified scoring metric, in this case, the F1-score. The F1-score was selected as the primary evaluation metric during grid search mainly because the Kaggle contest scoring is based on F1 but also because of its balance between precision and recall.

Models explored:

1.  **Random Forest Classifier (RF):** An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. It is robust to overfitting and can handle a large number of features.

2.  **Support Vector Classifier (SVC):** A supervised learning model used for classification and regression analysis. It finds the optimal hyperplane that best separates data points of different classes in a high-dimensional space.

3.  **XGBoost:** An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.

4.  **Logistic Regression:** A statistical model that uses a logistic function to model a binary dependent variable. It's a linear model that estimates the probability of a binary outcome.

## Analysis of Other Model Performance

To provide context for the RNN models' performance, we also evaluated several traditional supervised learning models: Random Forest, SVC, XGBoost, and Logistic Regression. We will compare their performance against the general performance observed across the RNN models.

First, considering the Random Forest model, its performance as shown in the "Supervised Model Performance Comparison Across Metrics" bar plot indicates lower scores across all metrics (Accuracy, F1, Precision, Recall) compared to most of the RNN models. The confusion matrix for Random Forest highlights a significant number of false negatives, indicating that it struggles to correctly identify disaster tweets. This suggests that while Random Forest is a powerful algorithm, the nature of the tweet data, with its sequential and contextual dependencies, might not be optimally captured by this model compared to RNNs.

Next, the Support Vector Classifier (SVC) demonstrates noticeably better performance than Random Forest, with higher scores across all metrics. Observing the bar plot, SVC's F1-score and Accuracy are closer to the lower end of the RNN models' performance range. Its confusion matrix shows a more balanced distribution of errors compared to Random Forest, but it still has a considerable number of false negatives and false positives when compared to the better-performing RNNs. SVC, being a non-sequential model, might not fully leverage the ordering of words in a tweet, which is where RNNs excel.

The XGBoost model, a gradient boosting algorithm, also shows improved performance compared to Random Forest. However, its F1-score and Recall are still generally lower than the majority of the RNN models, as seen in the bar plot. The XGBoost confusion matrix reveals a relatively high number of false negatives, similar to Random Forest, suggesting difficulty in capturing all positive instances. While XGBoost is known for its strong performance on various tabular datasets, it may not be as well-suited for sequential text data compared to architectures specifically designed for it, like RNNs.

Finally, Logistic Regression, a linear model, shows performance comparable to SVC and XGBoost, with an F1-score and Accuracy within the mid-range of the supervised models but generally lower than the RNNs. Its confusion matrix shows a similar pattern of errors to SVC. As a linear model, Logistic Regression's ability to capture complex, non-linear relationships and dependencies in the text data might be limited compared to deep learning models.

Comparing the supervised models to the RNN models generally, the RNN models exhibit superior performance across most evaluation metrics, particularly F1-score and AUC. This suggests that deep learning architectures, specifically designed to handle sequential data, are better equipped to capture the intricate patterns and dependencies within tweet text for this classification task.

Among the traditional supervised models, the SVC model appears to be the best performing based on the F1-score (0.7455). However, comparing this to the best performing RNN model, the BiLSTM (F1-score of 0.7715), the BiLSTM model still holds an advantage. The BiLSTM's ability to process sequences bidirectionally and capture long-range dependencies likely contributes to this superior performance on the tweet classification task.

![image](https://github.com/user-attachments/assets/41f968af-1764-4080-96a0-aaa50589713d)
![image](https://github.com/user-attachments/assets/fd4d77ff-0d6f-4789-9c65-8ac86cf6642e)
![image](https://github.com/user-attachments/assets/ac86f646-cc3e-4632-890b-c0e511ad74b2)
![image](https://github.com/user-attachments/assets/4eb4cea8-0ca7-4754-bac0-51bbf1b131d2)

# Conclusion

In this project, we focused on classifying tweets as either disaster-related or not using the Kaggle "Natural Language Processing with Disaster Tweets" dataset. Following initial data preparation, we extensively explored various deep learning Recurrent Neural Network (RNN) architectures and compared their performance to traditional supervised learning models.

Our analysis of the RNN models revealed a range of effectiveness, with the BiLSTM architecture demonstrating superior performance based on the F1-score and AUC metrics. This success is likely attributable to its bidirectional nature, which allows the model to capture contextual information from both preceding and succeeding words in a tweet, crucial for understanding nuances in natural language.

Comparing the RNN models to traditional supervised learning methods like Random Forest, SVC, XGBoost, and Logistic Regression highlighted the advantage of deep learning for this sequential text classification task. The RNN models, as a group, generally outperformed the traditional models, indicating their better ability to learn complex patterns and dependencies within the tweet data. While the SVC model was the best among the traditional methods, the leading RNN model, BiLSTM, still achieved a higher F1-score.

Despite achieving a respectable score on the Kaggle leaderboard, indicating good generalization to unseen data, there are limitations and areas for improvement. Our preprocessing, while effective, could be further refined, for instance, by exploring more advanced techniques for handling emojis or slang. In addition, while removing words that are common to both classes improved performance, it does result in a loss of information that could impact performance on unseen data.

Future work could involve experimenting with more sophisticated deep learning architectures, such as Transformer networks, which have shown state-of-the-art results in various NLP tasks. Additionally, incorporating external data sources or using pre-trained language models like BERT could potentially boost performance. Further hyperparameter tuning and cross-validation on a larger scale could also lead to some gains although extensive hyperparameter tuning using these architectures resulted in only marginal gains. Addressing the class imbalance with more advanced techniques than class weighting, might also yield improvements.
