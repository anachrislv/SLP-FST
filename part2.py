##################### imports #####################
import re
import contractions
from scripts.helpers import run_cmd


from gensim.models import Word2Vec

from gensim.models.keyedvectors import KeyedVectors
import warnings
warnings.simplefilter("ignore")

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

##################### step 12 #####################

def clean_text(s):     #taken from fetch_gutenberg script
    s = s.strip()  # Strip leading / trailing spaces
    s = s.lower()  # Convert to lowercase
    s = contractions.fix(s)  # e.g. don't -> do not, you're -> you are
    s = re.sub("\s+", " ", s)  # Strip multiple whitespace
    s = re.sub(r"[^a-z\s]", " ", s)  # Keep only lowercase letters and spaces

    return s

run_cmd("python scripts/fetch_gutenberg_alt.py > data/corpus_alt.txt")
infile = open("data/corpus_alt.txt")
corpus = infile.read()

tok = sent_tokenize(corpus)


# we faced a bug when we used the original script, so we use this method in this case
clean = [clean_text(s) for s in tok]

words = [word_tokenize(s) for s in clean]


model = Word2Vec(words, window=5, workers=8)    #default size is 100
model.train(words, total_examples=len(tok), epochs=1000)
model.save("w2v1000.model")

#changing epochs
model_2000 = Word2Vec(words, window=5, workers=8)
model_2000.train(words, total_examples=len(tok), epochs=2000)
model_2000.save("w2v2000.model")

model100 = Word2Vec(words, window=5, workers=8)
model100.train(words, total_examples=len(tok), epochs=100)
model100.save("w2v100.model")

model10 = Word2Vec(words, window=5, workers=8)
model10.train(words, total_examples=len(tok), epochs=10)
model10.save("w2v10.model")

#changing window for 100 epochs
model100win = Word2Vec(words, window=10, workers=8)
model100win.train(words, total_examples=len(tok), epochs=100)
model100win.save("w2v100win10.model")


list1 = ["bible", "book", "bank", "water"]

# find similar words using cosine similarity
# after reading the documentation we fount that this finction
# is implemented as most_similar()
def cosine_sim(model):
    for word in list1:
        similar = model.wv.most_similar(word, topn=5)
        print("Similar words for:", word, similar,"\n")


model = Word2Vec.load("w2v1000.model")

model2000 = Word2Vec.load("w2v2000.model")
model100 = Word2Vec.load("w2v100.model")
model10 = Word2Vec.load("w2v10.model")
model100win = Word2Vec.load("w2v100win10.model")

print("1000 epoch model:\n")
cosine_sim(model)
print("2000 epoch model:\n")
cosine_sim(model2000)
print("100 epoch model:\n")
cosine_sim(model100)
print("100 epoch model, larger window:\n")
cosine_sim(model100win)
print("10 epoch model:\n")
cosine_sim(model10)

list2 = [["girls", "queen", "kings"], ["good", "tall", "taller"], ["france", "paris", "london"]]

# compute most similar world with vector algebra and cosine similarity
def cosine_sim_compute_word(model):
    for w1, w2, w3 in list2:
        print(f"Word trio: ({w1}, {w2}, {w3})")
        similar = model.wv.most_similar(positive=[w1, w3], negative=[w2], topn = 5)
        print(f"Most similar words for {w1} - {w2} + {w3}: {similar}\n")

print("1000 epoch model:")
cosine_sim_compute_word(model)

#load google's model
googlemodel = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True, limit=350000)

#check results
print("Google model:")
cosine_sim(googlemodel)
cosine_sim_compute_word(googlemodel)


##################### step 13 #####################

#tensorflow tool results shown in report
with open("data/embeddings.tsv", "w") as infile:
    for word in model.wv.vocab.keys():
        for vector in list(model.wv.get_vector(word)):
            print(vector, file=infile, end="\t")
        print(file=infile)

with open("data/metadata.tsv", "w") as infile:
    for word in model.wv.vocab.keys():
        print(word, file=infile)

##################### step 14 #####################

#train and evaluate using w2v_sentiment_analysis.py the script
run_cmd("python scripts/w2v_sentiment_analysis.py")
