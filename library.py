import re
import string
import typing

import nltk
import numpy
import sklearn
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from tensorflow import keras
from transformers import T5Tokenizer, T5ForConditionalGeneration

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

MODEL_NAME = 't5-large'
SEED = 42


def embedding_for_vocab(filepath: str, word_index: dict, embedding_dim: int) -> numpy.ndarray:
    """
    Creates an embedding matrix for a given vocabulary using pre-trained word embeddings.

    :param filepath: The path to the pre-trained embedding file (e.g., GloVe or Word2Vec format).
                     Each line in the file should contain a word followed by its embedding vector.
    :param word_index: A dictionary mapping words in the vocabulary to their unique integer indices.
    :param embedding_dim: The dimensionality of the word embeddings.
    :return: A numpy array representing the embedding matrix, where each row corresponds to a word in the vocabulary.
             Words not found in the pre-trained embeddings will have a row of zeros.
    """
    vocab_size = len(word_index) + 1

    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = numpy.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = numpy.array(vector, dtype=numpy.float32)[:embedding_dim]

    return embedding_matrix_vocab


def embed_texts(texts: list, tokenizer: keras.preprocessing.text.Tokenizer,
                embedding_matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Converts a list of texts into their corresponding embedding representations
    by summing the embeddings of the words in each text.

    :param texts: A list of input strings (texts) to be converted into embeddings.
    :param tokenizer: A Keras tokenizer object used to convert texts to sequences of word indices.
    :param embedding_matrix: A numpy array representing the embedding matrix, where each row corresponds to a word's embedding vector.
    :return: A numpy array where each row is the summed embedding vector of the corresponding input text.
    """
    # Convert texts to sequences of word indices using the tokenizer
    text_data_seq = tokenizer.texts_to_sequences(texts=texts)

    # Initialize an array to hold the text embeddings
    x = numpy.zeros(shape=(len(texts), embedding_matrix.shape[1]))

    # Compute the embedding for each text by summing the embeddings of its words
    for j in range(len(text_data_seq)):
        for i in range(len(text_data_seq[j])):
            x[j] += embedding_matrix[text_data_seq[j][i]]

    return x


def gaussian_naive_bayes_trainer(
        model_path: str, tokenizer: keras.preprocessing.text.Tokenizer, embedding_dim: int, texts: typing.List[str],
        target: numpy.ndarray, cv: int = 10, random_state: int = None, test_size: float = 0.2
) -> typing.Dict[str, typing.Union[float, typing.Any]]:
    """
    Trains a Gaussian Naive Bayes classifier on text data and evaluates it using cross-validation and test metrics.
    The function creates an embedding matrix based on a pretrained model, transforms texts into embeddings, and uses
    these embeddings as features for classification. It returns a dictionary with evaluation scores, the trained model,
    and the embedding matrix.

    :param model_path: Path to the pretrained embedding model file used to create the embedding matrix.
    :param tokenizer: Keras Tokenizer used for mapping words to indices in the vocabulary.
    :param embedding_dim: Dimensionality of the word embeddings.
    :param texts: List of text samples to be embedded and used as input features.
    :param target: Array containing target labels for each text sample.
    :param cv: Number of cross-validation folds (default is 10).
    :param random_state: Seed for random number generator for reproducibility (default is None).
    :param test_size: Proportion of the data to include in the test split (default is 0.2).

    :return: A dictionary with the following keys:
        - 'cv_score': Mean balanced accuracy score from cross-validation.
        - 'test_accuracy': Balanced accuracy score on the test set.
        - 'fpr': False positive rate values for the ROC curve.
        - 'tpr': True positive rate values for the ROC curve.
        - 'roc_auc': Area under the ROC curve (AUC).
        - 'model': Trained Gaussian Naive Bayes model.
        - 'embedding_matrix': Embedding matrix used to transform text samples.
    """
    gnb_dict = {}

    # matrix for vocab: word_index
    embedding_matrix_vocab = embedding_for_vocab(
        filepath=model_path,
        word_index=tokenizer.word_index,
        embedding_dim=embedding_dim
    )

    # store the feature matrix (x) and response vector (y)
    x = embed_texts(
        texts=texts,
        tokenizer=tokenizer,
        embedding_matrix=embedding_matrix_vocab
    )

    y = target

    gnb = GaussianNB()

    gnb_dict['cv_score'] = numpy.average(a=cross_val_score(
        estimator=gnb,
        X=x,
        y=y,
        scoring='balanced_accuracy',
        cv=cv
    ))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state
    )

    gnb.fit(X=x_train, y=y_train)
    y_pred = gnb.predict(X=x_test)
    fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)

    gnb_dict['test_accuracy'] = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    gnb_dict['fpr'] = fpr
    gnb_dict['tpr'] = tpr
    gnb_dict['roc_auc'] = auc(x=fpr, y=tpr)
    gnb_dict['model'] = gnb
    gnb_dict['embedding_matrix'] = embedding_matrix_vocab

    return gnb_dict


def clean_tweet(tweet: str) -> str:
    """
    Cleans a tweet by removing retweet headers, mentions, and URLs.

    :param tweet: The original tweet text as a string.
    :return: The cleaned tweet text with retweet markers, mentions, and URLs removed.
    """
    tweet = re.sub(
        pattern=r'^RT @\w+: ',
        repl='',
        string=tweet
    )

    tweet = re.sub(
        pattern=r'@(\w+)',
        repl=r'\1',
        string=tweet
    )

    tweet = re.sub(
        pattern=r'https?://\S+|www\.\S+',
        repl='',
        string=tweet
    )

    return tweet.strip()


def lowercase(s: str) -> str:
    """
    Converts a given string to lowercase.

    :param s: The input string to be converted to lowercase.
    :return: A new string with all characters in lowercase.
    """
    return s.lower()


def remove_punctuation(s: str) -> str:
    """
    Removes all punctuation characters from a given string.

    :param s: The input string from which punctuation will be removed.
    :return: A new string with all punctuation characters removed.
    """
    return s.translate(str.maketrans('', '', string.punctuation))


def remove_whitespaces(s: str) -> str:
    """
    Removes extra whitespace from a given string, leaving only single spaces between words.

    :param s: The input string from which to remove extra whitespace.
    :return: A new string with extra whitespace removed, containing only single spaces between words.
    """
    return " ".join(s.split())


def remove_html_tags(s: str) -> str:
    """
    Removes HTML tags from a given string, returning only the text content.

    :param s: The input string containing HTML content.
    :return: A new string with HTML tags removed, preserving only the text content.
    """
    soup = BeautifulSoup(s, "html.parser")
    stripped_input = soup.get_text(separator=" ")

    return stripped_input


def tokenize(s: str) -> list:
    """
    Tokenizes a given string into a list of words.

    :param s: The input string to be tokenized.
    :return: A list of tokens (words) extracted from the input string.
    """
    return word_tokenize(s)


def remove_stop_words(s: str) -> typing.List[str]:
    """
    Removes English stop words from a given string, returning a list of the remaining words.

    :param s: The input string from which to remove stop words.
    :return: A list of words from the input string with stop words removed.
    """
    s = word_tokenize(s)
    return [word for word in s if word not in stopwords.words('english')]


def lemmatize(s: str) -> str:
    """
    Lemmatizes each word in the given string, reducing words to their base form.

    :param s: The input string to be lemmatized.
    :return: A new string with all words lemmatized to their base form.
    """
    lemmatizer = WordNetLemmatizer()
    input_str = word_tokenize(s)
    new_words = []

    for word in input_str:
        new_words.append(lemmatizer.lemmatize(word))

    return ' '.join(new_words)


def nlp_pipeline(s: str) -> str:
    """
    Processes a string through a series of NLP steps:
    converting to lowercase, removing HTML tags, punctuation, extra whitespaces,
    removing stop words, and lemmatizing the words.

    :param s: The input string to be processed.
    :return: A processed string after applying all NLP steps.
    """
    return lemmatize(' '.join(
        remove_stop_words(remove_whitespaces(remove_punctuation(remove_html_tags(lowercase(s)))))
    ))


def find_topics(question_body: str) -> ndarray:
    """
    Identifies topics in the given question body using Latent Dirichlet Allocation (LDA) after preprocessing the text.
    The text is first processed through an NLP pipeline, and then LDA is applied to extract the topics.

    :param question_body: The input string (question body) for which topics are to be identified.
    :return: An array of topics extracted from the question body. Each topic is represented as a list of words.
    """
    try:
        text = nlp_pipeline(question_body)
        count_vectorizer = CountVectorizer(stop_words='english')
        count_data = count_vectorizer.fit_transform([text])

        # One topic that has an avg of two words because most questions had 1/2 tags
        number_topics = 1
        number_words = 2

        # Create and fit the LDA model
        lda = LatentDirichletAllocation(n_components=number_topics, n_jobs=-1)
        lda.fit(count_data)

        words = count_vectorizer.get_feature_names_out()

        # Get topics from model. They are represented as a list e.g. ['military','army']
        topics = [[
            words[i] for i in topic.argsort()[:-number_words - 1:-1]
        ] for (topic_idx, topic) in enumerate(lda.components_)]

        topics = numpy.array(topics).ravel()

        # Only use topics for which a tag already exists
        # existing_topics = set.intersection(set(topics), unique_tags)

    # Three question bodies don't work with LDA so this exception just ignores them
    except ValueError:
        # return numpy.array(question_body).ravel()
        return numpy.array('').ravel()

    return topics


def dbscan_predict(dbscan: sklearn.cluster.DBSCAN, x: numpy.ndarray) -> numpy.ndarray:
    """
    Predicts the cluster labels for a new set of data points using a pre-trained DBSCAN model.

    This function calculates the distance of each new data point to the DBSCAN model's core samples
    and assigns a cluster label based on the nearest core sample, if it is within the model's epsilon distance.

    :param dbscan: The trained DBSCAN model used for prediction. It contains core sample information and clustering parameters.
    :param x: The new input data points, represented as a numpy array, for which cluster labels are to be predicted.
    :return: A numpy array of predicted cluster labels, where -1 indicates noise (unclassified points) and other integers represent cluster labels.
    """
    nr_samples = x.shape[0]
    y_new = numpy.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = dbscan.components_ - x[i, :]
        dist = numpy.linalg.norm(diff, axis=1)
        shortest_dist_idx = numpy.argmin(dist)

        if dist[shortest_dist_idx] < dbscan.eps:
            y_new[i] = dbscan.labels_[dbscan.core_sample_indices_[shortest_dist_idx]]

    return y_new


def complete_sentence(prompt: str) -> str:
    """

    :param prompt:
    :return:
    """
    # Prepend "fill: " to indicate the task to T5 (this is how T5 expects tasks to be framed)
    input_text = f'fill: {prompt}'

    # Tokenize the input text
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    # Generate the completion
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    output_ids = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        temperature=.7
    )

    # Decode and return the output text
    completed_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return completed_text


def projected_clustering(data) -> KMeans:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # Dimensionality reduction using PCA
    np.random.seed(seed=SEED)
    num_selected_dimensions = 2
    pca = PCA(n_components=num_selected_dimensions)
    projected_data = pca.fit_transform(X=data)

    # Perform k-means clustering on the projected data
    number_of_clusters = 3
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=SEED)
    kmeans.fit(X=projected_data)
    cluster_labels = kmeans.labels_

    # Plot the clusters
    fig, ax = plt.subplots()
    ax.scatter(projected_data[:, 0], projected_data[:, 1], c=cluster_labels)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PC1 vs. PC2')
    plt.show()

    return kmeans
