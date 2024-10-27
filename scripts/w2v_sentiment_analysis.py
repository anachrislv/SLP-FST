from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import glob
import numpy as np
import os
import re
import sklearn
from sklearn.linear_model import LogisticRegression

#from sklearnex import patch_sklearn
#patch_sklearn()


SCRIPT_DIRECTORY = os.path.dirname(__file__)

data_dir = os.path.join(SCRIPT_DIRECTORY, "../aclImdb/")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 10000 #had more ram
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 350000


SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)


def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)


def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())


def tokenize(s):
    return s.split(" ")


def preproc_tok(s):
    return tokenize(preprocess(s))


def pre(s):
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = s.lower()
    s = re.sub("\s+", " ", s)
    return s


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []

    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, "r", encoding="utf8") as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)

    return data


def create_corpus(pos, neg):    
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])

# we tokenize the words and the reviews and get the word embeddings
# using our trained models
# we then get the mean for its review to create the final vector
# and use it to train the Logistic Regression model
def extract_nbow(corpus, size, model):
    nbow = []

    for review in corpus:
        counter = 0
        vec = np.zeros(size)
        setmap = set(model.wv.index2word)  #index2word since gensim is version 3.8
        for word in review:
            counter = counter + 1
            if word in setmap:
                vec = np.add(vec, model[word])
        vec = np.divide(vec, counter)
        nbow.append(vec)

    return nbow


def train_sentiment_analysis(train_corpus, train_labels):
    """Train a sentiment analysis classifier using NBOW + Logistic regression"""
    clf = LogisticRegression(random_state=SEED)
    clf.fit(train_corpus, train_labels)
    return clf


def evaluate_sentiment_analysis(classifier, test_corpus, test_labels):
    """Evaluate classifier in the test corpus and report accuracy"""
    accuracy = sklearn.metrics.accuracy_score(
        test_labels, classifier.predict(test_corpus)
    )
    print(f"Accuracy: {accuracy:.3f}")
    return accuracy


if __name__ == "__main__":
    
    # load models
    model = Word2Vec.load( "../w2v1000.model")
    
    google_model = KeyedVectors.load_word2vec_format(
        "../GoogleNews-vectors-negative300.bin", binary=True, limit = NUM_W2V_TO_LOAD
    )

    # read pos/neg data and split in train-test
    # we did not have to use train-test split
    pos_train = read_samples(pos_train_dir, preproc_tok)
    neg_train = read_samples(neg_train_dir, preproc_tok)
    pos_test = read_samples(pos_test_dir, preproc_tok)
    neg_test = read_samples(neg_test_dir, preproc_tok)

    # create corpora for train and test data
    train_corpus, train_labels = create_corpus(pos_train, neg_train)
    test_corpus, test_labels = create_corpus(pos_test, neg_test)
    
    # evaluate our model
    nbow_model_train = extract_nbow(train_corpus, 100, model)
    log_model = train_sentiment_analysis(nbow_model_train, train_labels)

    nbow_model_test = extract_nbow(test_corpus, 100, model)
    evaluate_sentiment_analysis(log_model, nbow_model_test, test_labels)
    
    #evaluate google model
    nbow_google_train = extract_nbow(train_corpus, 300, google_model)
    log_google = train_sentiment_analysis(nbow_google_train, train_labels)
    #Accuracy: approx 74.4%

    nbow_google_test = extract_nbow(test_corpus, 300, google_model)
    evaluate_sentiment_analysis(log_google, nbow_google_test, test_labels)
    #Accuracy: approx 84.1%
