import numpy as np
import pickle
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

RESOURCE_PATH = {
    'WORD_EMBEDDINGS': 'starspace_embedding.tsv',
    'QUESTION_MATRIX': 'question_embeddings',
    'ANSWER_PAIRS': 'answer_list.txt'
}

def text_prepare(text):
    """
    Performs tokenization and simple preprocessing.
    """   
#     replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
#     bad_symbols_re = re.compile('[^0-9a-z #+_]')
#     stopwords_set = set(stopwords.words('english'))
#     tags_re = re.compile('<[^>]*>')
#     arrow_re = re.compile('[<>]')

#     text = text.lower()
#     text = replace_by_space_re.sub(' ', text)
#     text = bad_symbols_re.sub('', text)
#     text = tags_re.sub('', text)
#     text = arrow_re.sub('', text)
#     text = ' '.join([x for x in text.split() if x and x not in stopwords_set])
    GOOD_SYMBOLS_RE = re.compile('[^0-9a-z ]')
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#+_]')
    REPLACE_SEVERAL_SPACES = re.compile('\s+')

    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = GOOD_SYMBOLS_RE.sub('', text)
    text = REPLACE_SEVERAL_SPACES.sub(' ', text)    

    return text.strip()

def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    embeddings = {}
    for line in open(embeddings_path):
        word, *arr = line.split('\t')
        embeddings[word] = np.asarray(arr, dtype='float32')
        
    dim = len(arr)
    
    return embeddings, dim

def question_to_vec(question, embeddings, dim):
    """
    Transforms a string to an embedding by averaging word embeddings.
    """    
    question2vec = [embeddings[word] for word in question.split() if word in embeddings]
    
    if not question2vec:
        return np.zeros(dim)
    
    question2vec = np.array(question2vec)
    
    return question2vec.mean(axis=0)
    
def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)    