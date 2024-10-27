######################## imports ########################

import numpy as np
from scripts.util import format_arc, EPS, CHARS, INFINITY
from scripts.helpers import run_cmd
from collections import Counter

######################## Step 11 ########################

def compose_fsts(A, B, C):
    run_cmd(f"fstarcsort --sort_type=olabel fsts/{A}.binfst fsts/{A}_ofstarcsort.binfst")
    run_cmd(f"fstarcsort --sort_type=ilabel fsts/{B}.binfst fsts/{B}_ifstarcsort.binfst")
    run_cmd(f"fstcompose fsts/{A}_ofstarcsort.binfst fsts/{B}_ifstarcsort.binfst fsts/{C}.binfst")


def get_freq(path):
    with open(path) as infile:
        freq_dict = Counter([tuple(line.strip().split("\t")) for line in infile.readlines()])
        total_edits = sum(freq_dict.values())
        # apply Add-1 smoothing for zero frequency edits
        for edit in [(a,b) for a in CHARS + [EPS] for b in CHARS + [EPS]]:
            if edit not in freq_dict:
                freq_dict[edit] = 1
                total_edits += 1
        return {edit: (freq+1)/total_edits for edit, freq in freq_dict.items()}


freq_dict = get_freq("data/edits.txt")

edits = sum(i for _, i in freq_dict.items())

# get the negative logarithm of the freqs
edit_dict11 = {
    (i, j): -np.log(freq)
    for ((i, j), freq) in freq_dict.items()
}



def create_E(chars, path, dictionary):
    with open(path, "w") as infile:
        for c in chars:
            print(format_arc(0, 0, c, c, weight=0), file=infile)
            key = (c, EPS)
            w = dictionary.get(key, INFINITY)
            print(format_arc(0, 0, c, EPS, weight=w), file=infile)
            key = (EPS, c)
            w = dictionary.get(key, INFINITY)
            print(format_arc(0, 0, EPS, c, weight=w), file=infile)
            for c2 in chars:
                if c == c2:
                    continue
                key = (c, c2)
                w = dictionary.get(key, INFINITY)
                print(format_arc(0, 0, c, c2, weight=w), file=infile)
        print(0, file=infile)


create_E(CHARS, "fsts/E_step11.fst", edit_dict11)

run_cmd(
    "fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/E_step11.fst fsts/E_step11.binfst"
)

compose_fsts("E_step11", "V", "E11V")

run_cmd("python scripts/run_evaluation.py fsts/E11V.binfst > outputs/E11V_evaluation.txt")


# we find that the accuracy has improved from 0.688 to 0.6925


################################################################################################################################

# Since the preparation for the second part of this step contained many different scripts, we have saved the mandatory files
# in the data folder and provide some explanatory comments for the preprocessing steps

"""
# run a modification of fetch_gutenberg that creates corpus from the link

# this downloads the 40k most frequently found words, sorted
run_cmd(
   "python scripts/fetch_url.py https://raw.githubusercontent.com/dolph/dictionary/master/popular.txt > data/corp.txt"
)

# this uses the most frequent words from the subtitles. 

# These corpora are contaminated, containing non existent, or grammatically incorrect words (e.g "aaaaah" or "bycicle")

run_cmd(
   "python scripts/fetch_url.py https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/en/en_50k.txt > data/corp1.txt"
)
run_cmd(
   "python scripts/fetch_url.py https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/en/en_full.txt > data/corp2.txt"
)

#the second one contains approx. 500k words
#we use the following code to filter the words that appear less than five times (although this is a small threshold)

file_path = "data/corp1.txt"  # replace with your file path
stop_word = "madmax"  # this is our stop word

with open(file_path, "r") as file:
    lines = file.readlines()

modified_lines = []
for line in lines:
    if stop_word in line:
        modified_lines.append(line)
        break  # stop iterating once the stop word is found
    modified_lines.append(line)

with open(file_path, "w") as file:
    file.writelines(modified_lines)
"""

# we convert words.syms to words.txt
# we use "cat corp${number_placeholder}.txt words.txt | sort | uniq > corp.txt" to merge with gutenberg corpora
# we use sed 's/[0-9]*//g' corp3.txt> corp4.txt to only keep the words without indexes

#we delete <eps> because we will add it in next steps
# now we have corpura of approx 41k words, 65k words and 250k words

# we use the same steps
# corp4 -> 41k
# corp1 -> 65k
# corp2 -> 250k

with open("data/corp4.txt") as infile:
    dict40 = dict(Counter(infile.read().split()))


with open("vocab/words40.syms", "w") as infile:
    print(f"{EPS}\t0", file=infile)
    i = 1
    for word in dict40:
        print(f"{word}\t{i}", file=infile)
        i += 1

with open("data/corp1.txt") as infile:
    dict65 = dict(Counter(infile.read().split()))


with open("vocab/words65.syms", "w") as infile:
    print(f"{EPS}\t0", file=infile)
    i = 1
    for word in dict65:
        print(f"{word}\t{i}", file=infile)
        i += 1

with open("data/corp2.txt") as infile:
    dict250 = dict(Counter(infile.read().split()))


with open("vocab/words250.syms", "w") as infile:
    print(f"{EPS}\t0", file=infile)
    i = 1
    for word in dict250:
        print(f"{word}\t{i}", file=infile)
        i += 1

# create an acceptor V 
# V accepts every word from the created dictionary
def create_V(dict, path):
    s = 1
    accept_state = sum([len(word) for word in dict]) + 1

    with open(path, "w") as infile:
        for word in dict:
            for i, char in enumerate(word):
                if i == 0:
                    # fist letter but we get the whole word
                    print(format_arc(0, s, char, word, weight=0), file=infile)
                else:
                    # use <eps> for the next letters
                    print(format_arc(s, s + 1, char, EPS, weight=0), file=infile)
                    s += 1

                if i == len(word) - 1:
                    # reaching the final letter we connect with accepting state (using <eps>)
                    print(
                        format_arc(s, accept_state, EPS, EPS, weight=0),
                        file=infile,
                    )

            s += 1

        print(accept_state, file=infile)

# use on the filtered dictionary
create_V(dict40, "fsts/V40_first.fst")
create_V(dict65, "fsts/V65_first.fst")
create_V(dict250, "fsts/V250_first.fst")

# we use different versions of the run evaluation and predict scripts for the different symbols, so that we do not have to pass the file as an argument
for vocab_size in [40, 65, 250]:
    run_cmd(
        f"fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words{vocab_size}.syms fsts/V{vocab_size}_first.fst  fsts/V{vocab_size}_first.binfst"
    )
    run_cmd(
        f"fstrmepsilon fsts/V{vocab_size}_first.binfst | fstdeterminize | fstminimize > fsts/V{vocab_size}.binfst"
    )
    run_cmd(
        f"fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words{vocab_size}.syms fsts/V{vocab_size}.binfst > fsts/V{vocab_size}.fst"
    )
    run_cmd(
        f"fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words{vocab_size}.syms fsts/V{vocab_size}.fst fsts/V{vocab_size}.binfst"
    )
    compose_fsts("E_step11", f"V{vocab_size}", f"E11V{vocab_size}")
    run_cmd(f"python scripts/run_ev{vocab_size}.py fsts/E11V{vocab_size}.binfst > outputs/E11V{vocab_size}_evaluation.txt")

# after evaluating the 40k word corpus we find that the accuracy has increased from 0.6925 to 0.8
# the 65k word corpus worsened the accuracy, achieving 0.688 just like the original EV transducer and the 250k word corpus
# dropped it down to 0.474. This shows how important the quality of our dictionary is, since in both cases where the dictionaries
# were polluted, we achieved worse resaults, even when their sizes were much greater.
