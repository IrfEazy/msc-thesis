import ast
import re

import emoji
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize


def clean_text(text):
    # Remove unnecessary characters
    text = re.sub(r'[^\w\s]', '', text)
    # Replace repetitive line breaks and blank spaces with only one
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove emoticons and emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    return text


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


def pos_tagging(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


def extract_structural_features(text):
    # Implement feature extraction logic
    message_length = len(text)
    num_tokens = len(word_tokenize(text))
    num_hashtags = text.count('#')
    num_emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    num_urls = text.count('http://') + text.count('https://')
    num_periods = text.count('.')
    num_commas = text.count(',')
    num_digits = sum(c.isdigit() for c in text)
    num_sentences = len(sent_tokenize(text))
    num_mentioned_users = text.count('@')
    num_uppercase = sum(c.isupper() for c in text)
    num_question_marks = text.count('?')
    num_exclamation_marks = text.count('!')
    emojis = set(re.findall(r':\w+:', emoji.demojize(text)))
    num_emoticons = len(emojis)
    num_dollar_symbols = text.count('$')
    # Other symbols
    num_other_symbols = len([
        char for char in text if
        char not in '"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@.://,?!' + ''.join(emojis)
    ])  # Return features as a list
    return [
        message_length,
        num_tokens,
        num_hashtags,
        num_emails,
        num_urls,
        num_periods,
        num_commas,
        num_digits,
        num_sentences,
        num_mentioned_users,
        num_uppercase,
        num_question_marks,
        num_exclamation_marks,
        num_emoticons,
        num_dollar_symbols,
        num_other_symbols
    ]


def replace_text_components(text):
    # Implement text component replacement logic,
    # For example, replace email addresses with 'email_nlp', replace mentioned users with 'at_user_nlp', etc.
    # Here's a simple example:
    text = text.replace('@', 'at_user_nlp')
    text = text.replace('#', '')  # Remove hashtags
    # Add more replacement rules as needed
    return text


def extract_first_url(url_list):
    try:
        urls = ast.literal_eval(url_list)
        first_url = urls[0] if urls else None
        return first_url
    except (SyntaxError, ValueError):
        return None


def extract_url_features(url, urls):
    # Extract domain suffix and registrant from the URL
    if pd.isna(url):
        return ['NA'] * 24  # Return NA for all features if URL is missing
    else:
        url_length = len(url)
        has_security_protocol = 'Y' if url.startswith(('http://', 'https://')) else 'N'
        # Feature 3 and 4: Creation date and Last update date (Days) - Not implemented
        # Extract the first URL from the list
        first_url = extract_first_url(urls)
        is_shortened_url = 'Y' if first_url and len(url) < len(first_url) else 'N'
        strings_divided_by_periods = len(url.split('.'))
        strings_divided_by_hyphens = len(url.split('-'))
        strings_divided_by_slashes = len(url.split('/'))
        num_words = len(re.findall(r'\b\w+\b', url))
        num_ips = len(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url))
        num_digits = sum(c.isdigit() for c in url)
        num_hyphens = url.count('-')
        num_periods = url.count('.')
        num_slashes = url.count('/')
        num_uppercase = sum(c.isupper() for c in url)
        num_lowercase = sum(c.islower() for c in url)
        num_ampersand_symbols = url.count('&')
        num_equal_symbols = url.count('=')
        num_question_marks = url.count('?')
        num_wave_symbols = url.count('~')
        num_plus_signs = url.count('+')
        num_colon_symbols = url.count(':')
        num_other_characters = len([
            char for char in url if
            char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@.://,?!-'
        ])
        has_extension = 'Y' if '.' in url else 'N'
        domain_suffix = url.split('.')[-1]
        registrant = url.split('/')[2].split('.')[-2] if '/' in url else 'NA'
        return [
            url_length,
            has_security_protocol,
            is_shortened_url,
            strings_divided_by_periods,
            strings_divided_by_hyphens,
            strings_divided_by_slashes,
            num_words,
            num_ips,
            num_digits,
            num_hyphens,
            num_periods,
            num_slashes,
            num_uppercase,
            num_lowercase,
            num_ampersand_symbols,
            num_equal_symbols,
            num_question_marks,
            num_wave_symbols,
            num_plus_signs,
            num_colon_symbols,
            num_other_characters,
            has_extension,
            domain_suffix,
            registrant
        ]


def replace_url_components(url):
    # Replace email addresses and mentioned users with predefined tokens
    replaced_url = re.sub(r'[\w.-]+@[\w.-]+', 'email_nlp', url)
    replaced_url = re.sub(r'@[\w.-]+', 'at_user_nlp', replaced_url)
    return replaced_url
