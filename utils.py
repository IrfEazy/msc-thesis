import json
import re
import typing
from pathlib import Path

import numpy
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import roc_curve, balanced_accuracy_score, auc
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from tensorflow import keras


def filter_dataset(input_file: Path, pattern: str, output_file: Path = None, inplace: bool = False) -> None:
    """
    Filter entries in a JSON dataset based on a specified pattern and save the filtered dataset.

    Parameters
    ----------
    input_file : Path
        Path to the input JSON file. The file must contain an "entries" key
        with a list of dictionaries, each representing an entry.
    pattern : str
        A regular expression pattern. Entries with a "body" field matching
        this pattern will be removed from the dataset.
    output_file : Path, optional
        Path to save the filtered JSON file. If not provided and `inplace` is
        True, the filtered dataset will overwrite the input file. If not provided
        and `inplace` is False, a `ValueError` will be raised. The file must
        have a `.json` extension.
    inplace : bool, default False
        If True, the input file will be overwritten with the filtered dataset.
        If False, a new file must be specified via `output_file`.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist or the output file cannot be found.
    TypeError
        If the input or output file is not a JSON file.
    ValueError
        If `output_file` is not specified and `inplace` is False.

    Notes
    -----
    - The function modifies the dataset by removing entries where the "body"
      field matches the given regular expression pattern.
    - If `inplace` is True, the input file is updated in place. Otherwise,
      a new file must be specified for the output.

    Examples
    --------
    Filter a dataset by removing entries containing the word "example" in the "body":

        >>> from pathlib import Path
        >>> filter_dataset(
        ...     input_file=Path("dataset.json"),
        ...     pattern=r"example",
        ...     output_file=Path("filtered_dataset.json")
        ... )

    Filter a dataset in place:

        >>> filter_dataset(
        ...     input_file=Path("dataset.json"),
        ...     pattern=r"example",
        ...     inplace=True
        ... )
    """
    output_file = input_file if output_file is None and inplace else output_file

    try:
        # Step 1: Load the JSON file
        with open(file=input_file) as file:
            extension = input_file.suffix

            if extension == ".json":
                data = json.load(fp=file)
            else:
                raise TypeError(f"File {input_file} is not JSON")

        # Step 2: Filter entries
        if "entries" in data:
            filtered_entries = [
                entry for entry in data["entries"]
                if entry.get("body") and not re.match(pattern=pattern, string=entry.get("body", ""))
            ]

            # Update the JSON structure
            data["entries"] = filtered_entries

        # Step 3: Save the result
        try:
            with open(file=output_file, mode="w") as file:
                extension = output_file.suffix

                if extension == ".json":
                    json.dump(obj=data, fp=file, indent=4)
                else:
                    raise TypeError(f"File {output_file} is not JSON")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {output_file} not found.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {input_file} does not exist")


def save_dataset_to_csv(input_file: Path, output_file: Path) -> None:
    """
    Convert a JSON dataset into a CSV file.

    Parameters
    ----------
    input_file : Path
        The path to the input JSON file containing the dataset. The JSON file should have
        an "entries" key that contains a list of dictionaries, where each dictionary represents a row.
    output_file : Path
        The path to save the resulting CSV file. If not specified, the input file will be overwritten.
        The file must have a `.csv` extension.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    TypeError
        If the input file is not a JSON file or the output file is not a CSV file.

    Notes
    -----
    - The function assumes that the JSON file contains an "entries" key, which holds a list of dictionaries.
    - The function writes the resulting DataFrame to the specified CSV file, with no index included.

    Examples
    --------
    Convert a JSON dataset to CSV:
        >>> from pathlib import Path
        >>> save_dataset_to_csv(Path("data.json"), Path("data.csv"))
    """
    output_file = input_file if output_file is None else output_file

    try:
        # Step 1: Load the JSON file
        with open(file=input_file) as file:
            extension = input_file.suffix

            if extension == ".json":
                data = json.load(fp=file)
            else:
                raise TypeError(f"File {input_file} is not JSON")

        # Step 2: Extract the 'entries' key
        entries = data.get("entries", [])

        # Step 3: Load entries into a DataFrame
        data = pd.DataFrame(data=entries)

        # Step 5: Save the DataFrame
        if output_file.suffix == ".csv":
            data.to_csv(path_or_buf=output_file, index=False)
        else:
            raise TypeError(f"File {output_file} is not CSV")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {input_file} not found.")


def extract_web_content(url: str) -> str:
    """Fetches and extracts the main text content from a web page."""
    try:
        # Fetch the webpage content using requests
        response = requests.get(url)
        response.raise_for_status()  # Ensure we get a successful response
        html_content = response.text

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style tags
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()  # Remove these tags from the soup

        # Extract and return the cleaned text content
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception:
        return ""


def extract_text_and_urls(text: str) -> list:
    """
    Given a text, returns a list of alternating segments:
    text without URLs and URLs.

    Parameters:
    text (str): The input string containing text and URLs.

    Returns:
    list: A list where each element is either a text segment or a URL.
    """
    # Regular expression to find URLs (http:// or https://)
    url_pattern = r'https?://[^\s\)\]\.,!?;]+(?:\.[^\s\)\]\.,!?;]+)+'
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)

    # Split the text by URLs to keep the non-URL parts
    non_url_parts = re.split(url_pattern, text)

    # Combine the non-URL parts and URLs into the result list
    result = []

    # Add non-URL parts and URLs alternately
    for non_url_part, url in zip(non_url_parts, urls):
        if non_url_part:  # if there is text before the URL
            result.append(non_url_part)
        result.append(url)  # append the URL

    # If there's any text left after the last URL, add it
    if non_url_parts[-1]:
        result.append(non_url_parts[-1])

    return result


def process_text_with_links(text: str) -> str:
    """
    Processes the given text, replacing URLs with their content and
    joining the list into a single string.

    Parameters:
    text (str): The input string containing text and URLs.

    Returns:
    str: The resulting string with URLs replaced by their content.
    """
    # Extract text and URLs
    extracted_parts = extract_text_and_urls(text)

    # Process each part: replace URLs with content
    processed_parts = []

    for part in extracted_parts:
        # If the part is a URL, replace it with the web content
        if part.startswith("http"):
            content = extract_web_content(part)
            processed_parts.append(content)
        else:
            processed_parts.append(part)

    # Join the list into a single string
    return ''.join(processed_parts)


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
