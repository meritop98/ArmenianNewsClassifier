import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from config import CFG

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def merge_dfs(files):
    """
    Merge multiple DataFrames from given files into a single DataFrame.

    Args:
        files (list): List of file paths.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged_df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(os.path.join(CFG.scraped_data_dir, file),
                         delimiter=',')  # Assuming the files are tab-separated (use appropriate delimiter)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        merged_df=merged_df[0:5]
    return merged_df


def translate_cols(df, col_names):
    """
    Translate specified columns of a DataFrame using Google Translate.

    Args:
        df (pd.DataFrame): DataFrame.
        col_names (list): List of column names to translate.

    Returns:
        pd.DataFrame: Translated DataFrame.
    """
    for col_name in col_names:
        df[f'{col_name}_processed'] = df[col_name].apply(
            lambda x: GoogleTranslator(source='auto', target='en').translate(x[:1000]))
    df = df.drop(col_names, axis=1)
    return df


def remove_punctuation(text):
    """
    Remove punctuation from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without punctuation.
    """
    punctuation_remover = text.maketrans("", "", string.punctuation)
    text_without_punctuations = text.translate(punctuation_remover)
    return text_without_punctuations


def remove_stop_words(text):
    """
    Remove stop words from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without stop words.
    """
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove the stop words from the tokenized words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def preprocess_data(files):
    # Merge the data from multiple files
    merged_df = merge_dfs(files)

    # Drop rows with missing values
    merged_df.dropna(inplace=True)

    # Translate the 'Title' and 'Text' columns
    translated_df = translate_cols(merged_df, ['Title', 'Text'])

    # Remove punctuation from 'Title' and 'Text' columns
    translated_df[['Title_processed', 'Text_processed']] = translated_df[['Title_processed', 'Text_processed']].applymap(remove_punctuation)

    # Remove stop words from 'Title' and 'Text' columns
    stop_words_removed = translated_df[['Title_processed', 'Text_processed']].applymap(remove_stop_words)
    result = stop_words_removed.rename(columns={"Title_processed": "Title",
                            "Text_processed": "Text"})

    # Save the preprocessed data to a CSV file
    return result
if __name__ == '__main__':
    import os
    # files = ['culture.csv', 'sport_.csv', 'it_.csv', 'politics_.csv', 'economy.csv',
    #          'covid.csv']
    # result = preprocess_data(files)
    # result.to_csv(os.path.join(CFG.data_dir, 'data_cleaned.csv'), index=False)



