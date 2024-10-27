######################## imports ########################

import numpy as np
from scripts.util import format_arc, EPS, CHARS, INFINITY
from scripts.helpers import run_cmd
from collections import Counter
import shutil

######################## Step 1 ########################

# run the script and save the downloaded corpus in data folder

run_cmd("python scripts/fetch_gutenberg.py > data/corpus.txt")

######################## Step 2 ########################


# create the dictionary
with open("data/corpus.txt") as infile:
    dictionary = dict(Counter(infile.read().split()))

# filter the dictionary based on threshold
filt_dict = dict(filter(lambda item: item[1] >= 5, dictionary.items()))

# create words.vocab.txt as instructed
with open("vocab/words.vocab.txt", "w") as infile:
    for i, j in filt_dict.items():
        print(f"{i}\t{j}", file=infile)


######################## Step 3 ########################

# map ascii to index
def chars_to_index(c):
    return 0 if c == EPS else ord(c)  # return c==eps ? 0 : ascii(c)


# create character symbols file
def chars_symbols():
    with open("vocab/chars.syms", "w") as infile:
        for c in [EPS] + CHARS:
            print(f"{c}\t{chars_to_index(c)}", file=infile)


chars_symbols()


# create word symbols file corrispondingly
def words_symbols(dict):
    with open("vocab/words.syms", "w") as infile:
        print(f"{EPS}\t0", file=infile)
        i = 1
        for word in dict:
            print(f"{word}\t{i}", file=infile)
            i += 1


words_symbols(filt_dict)


######################## Step 4 ########################

def create_L(chars, path):
    with open(path, "w") as infile:
        for c in chars:
            print(format_arc(0, 0, c, c, weight=0), file=infile)    #character to itself - no edit
            print(format_arc(0, 0, c, EPS, weight=1), file=infile)  # delete
            print(format_arc(0, 0, EPS, c, weight=1), file=infile)  # insert
            
            for c2 in chars:
                if c == c2:
                    continue
                print(format_arc(0, 0, c, c2, weight=1), file=infile)  # replace

        print(0, file=infile)  # accepting state


# save L in openfst text format
create_L(CHARS, "fsts/L.fst")

# compile L and save binary
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/L.fst fsts/L.binfst")

# follow the same steps to also draw a sample of chars with fstdraw
# this step requires installation of Graphviz and initialization of
# plugins to make .png format usable

draw_char = list("ab")

create_L(draw_char, "fsts/L_test.fst")

run_cmd(
    "fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/L_test.fst fsts/L_test.binfst"
)

run_cmd(
    "fstdraw --isymbols=vocab/chars.syms --osymbols=vocab/chars.syms -portrait fsts/L_test.binfst | dot -Tpng > "
    "fsts/L_test.png"
)


######################## Step 5 ########################

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
create_V(filt_dict, "fsts/V_first.fst")

# compile acceptor
run_cmd(
    "fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V_first.fst  fsts/V_first.binfst"
)

# optimize acceptor
run_cmd(
    "fstrmepsilon fsts/V_first.binfst | fstdeterminize | fstminimize > fsts/V.binfst"
)

# save in openfst text format
run_cmd(
    "fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V.binfst > fsts/V.fst"
)

# compile the optimized acceptor and save it
run_cmd(
    "fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V.fst fsts/V.binfst"
)

# use sample of words to draw an acceptor following the same steps
draw_dict = {i: j for i, j in list(filt_dict.items())[0:3]}

create_V(draw_dict, "fsts/V_first_test.fst")

run_cmd(
    "fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V_first_test.fst "
    "fsts/V_first_test.binfst"
)

run_cmd(
    "fstrmepsilon fsts/V_first_test.binfst fsts/V_rmepsilon_test.binfst"
)

run_cmd(
    "fstdeterminize fsts/V_rmepsilon_test.binfst fsts/V_rmepsilon_determinized_test.binfst"
)

run_cmd(
    "fstminimize fsts/V_rmepsilon_determinized_test.binfst fsts/V_test.binfst"
)

# draw test acceptor for every, before optimization, after fstrmepsilon, fstdeterminize and after fstminimize

for fstname in [
    "V_first_test",
    "V_rmepsilon_test",
    "V_rmepsilon_determinized_test",
    "V_test",
]:
    run_cmd(
        f"fstdraw --isymbols=vocab/chars.syms --osymbols=vocab/words.syms -portrait fsts/{fstname}.binfst | dot "
        f"-Tpng > fsts/{fstname}.png"
    )


######################## Step 6 ########################


# compose L with V to get S
def compose_fsts(A, B, C):
    """
    Use to compose A with B to get C

    This function uses fstarcsort (as instructed) to ensure optimization and correct composition

    A and B arguments are the names of the FST binaries used as input, without the extension
    C is the name of the output binary without the extension 

    """
    run_cmd(f"fstarcsort --sort_type=olabel fsts/{A}.binfst fsts/{A}_ofstarcsort.binfst")
    run_cmd(f"fstarcsort --sort_type=ilabel fsts/{B}.binfst fsts/{B}_ifstarcsort.binfst")
    run_cmd(f"fstcompose fsts/{A}_ofstarcsort.binfst fsts/{B}_ifstarcsort.binfst fsts/{C}.binfst")


compose_fsts("L", "V", "S")

######################## Step 7 ########################


# Test spell checker S
def test_S(input, S, output="/dev/stdout"):
    with open(input) as infile, open(output, "w") as outfile:
        for _ in range(20):
            words = infile.readline().strip().split(" ")
            correct = words[0][:-1] #remove ":" 
            incorrect = words[1:]
            for wrong in incorrect:
                corrected = run_cmd(
                    f"bash scripts/predict.sh {S} {wrong}"
                )
                print(
                    f"{wrong} --> {corrected}, correct word: {correct}",
                    file=outfile
                )


test_S("data/spell_test.txt", "fsts/S.binfst", "outputs/S_test.txt")

######################## Step 8 ########################


#word_edits.sh ${word} creates MLN for one word and executes fstshortestpath and fstprint
run_cmd("bash scripts/word_edits.sh abandonned abandoned")

#remove non English characters from wiki.txt
def is_latin(s):
    for c in s:
        # check if the character is a letter (A-Z, a-z)
        if "A" <= c <= "Z" or "a" <= c <= "z":
            continue
        # check if the character is a space, tab, or newline
        elif c in [" ", "\t", "\n"]:
            continue
        else:
            return False
    return True


# open the input and temporary files
with open("./data/wiki.txt", "r", encoding="utf-8") as infile, open("temp_file.txt", "w", encoding="utf-8") as temp_file:
    # loop through each line in the input file
    for line in infile:
        # split the line into two words separated by a tab
        word1, word2 = line.strip().split("\t")

        # check if either word contains non-Latin characters
        if not is_latin(word1) or not is_latin(word2):
            continue

        # write the line to the temporary file if both words contain only Latin characters
        temp_file.write(line)

# overwrite the original file with the contents of the temporary file
shutil.move("temp_file.txt", "./data/wiki.txt")

# save the edits in a file
with open("data/wiki.txt", "r") as infile, open("data/edits.txt", "w") as outfile:
    for line in infile:
        w, c = tuple(line.strip().split("\t"))
        output = run_cmd(f"bash scripts/word_edits.sh {w} {c}")
        print(output, file=outfile, end="")


# get frequency of each edit and save them in a dictionary
def get_freq(path):
    with open(path) as infile:
        return dict(
            Counter([tuple(line.strip().split("\t")) for line in infile.readlines()])
        )


freq_dict = get_freq("data/edits.txt")

edits = sum(i for _, i in freq_dict.items())

# get the negative logarithm of the freqs
edit_dict = {
    (i, j): -np.log(freq / edits)
    for ((i, j), freq) in freq_dict.items()
}

# we create transducer E similarly to L
# however we now use as weights -log(probability_of_edit(edit))
# or infinity if edit is not found
def create_E(chars, path, dictionary):
    with open(path, "w") as infile:
        for c in chars:
            print(format_arc(0, 0, c, c, weight=0), file=infile)
            key = (c, EPS)
            w = dictionary[key] if key in dictionary else INFINITY
            print(format_arc(0, 0, c, EPS, weight=w), file=infile)
            key = (EPS, c)
            w = dictionary[key] if key in dictionary else INFINITY
            print(format_arc(0, 0, EPS, c, weight=w), file=infile)
            for c2 in chars:
                if c == c2:
                    continue
                key = (c, c2)
                w = dictionary[key] if key in dictionary else INFINITY
                print(format_arc(0, 0, c, c2, weight=w), file=infile)
        print(0, file=infile)


# we save in OpenFst text format
create_E(CHARS, "fsts/E.fst", edit_dict)

# and we compile to get the E.binfst file
run_cmd(
    "fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/E.fst fsts/E.binfst"
)

# use small set of letters to draw E similarly to previous steps  
E_test_char = list("abc")
create_E(E_test_char, "fsts/E_test.fst", edit_dict)

run_cmd(
    "fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/E_test.fst fsts/E_test.binfst"
)
run_cmd(
    "fstdraw --isymbols=vocab/chars.syms --osymbols=vocab/chars.syms -portrait fsts/E_test.binfst | dot -Tpng >fsts/E_test.png"
)

# we use the compose function to create the spell checker EV and test it on the first twenty inputs of spell_test.txt
compose_fsts("E", "V", "EV")

test_S("data/spell_test.txt", "fsts/EV.binfst", "outputs/EV_test.txt")


######################## Step 9 ########################

#create W using the neg. loarithms extracted previously
def create_W(dictionary, path):
    with open(path, "w") as infile:
        total = sum(i for _, i in dictionary.items())
        for word, freq in dictionary.items():
            w = -np.log(freq / total)
            print(format_arc(0, 0, word, word, w), file=infile)
        print(0, file=infile)


create_W(filt_dict, "fsts/W_first.fst")


# Compile and optimize W
run_cmd(
    "fstcompile -isymbols=vocab/words.syms -osymbols=vocab/words.syms fsts/W_first.fst fsts/W_first.binfst"
)
run_cmd(
    "fstrmepsilon fsts/W_first.binfst | fstdeterminize | fstminimize > fsts/W.binfst"
)

# create LV and LVW
compose_fsts("L", "V", "LV")
compose_fsts("LV", "W", "LVW")

#create EV and EVW
compose_fsts("E", "V", "EV")
compose_fsts("EV", "W", "EVW")


test_S(
        "data/spell_test.txt", "fsts/LVW.binfst", "outputs/LVW_test.txt"
    )

create_V(draw_dict, "fsts/V_test.fst")
create_W(draw_dict, "fsts/W_test.fst")

run_cmd(
    "fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V_test.fst fsts/V_test.binfst"
)
run_cmd(
    "fstcompile -isymbols=vocab/words.syms -osymbols=vocab/words.syms fsts/W_test.fst fsts/W_test.binfst"
)

# Compose V with W to produce VW
compose_fsts("V_test", "W_test", "VW_test")

# Draw the resulting FSTs using a small subset of the words
run_cmd(
    "fstdraw --isymbols=vocab/words.syms --osymbols=vocab/words.syms -portrait fsts/W_test.binfst | dot -Tpng >fsts/W.png"
)
run_cmd(
    "fstdraw --isymbols=vocab/chars.syms --osymbols=vocab/words.syms -portrait fsts/VW_test.binfst | dot -Tpng >fsts/VW.png"
)

######################## Step 10 ########################

# evaluate the spell checkers
run_cmd("python scripts/run_evaluation.py fsts/LV.binfst > outputs/LV_evaluation.txt")
run_cmd("python scripts/run_evaluation.py fsts/LVW.binfst > outputs/LVW_evaluation.txt")
run_cmd("python scripts/run_evaluation.py fsts/EV.binfst > outputs/EV_evaluation.txt")
run_cmd("python scripts/run_evaluation.py fsts/EVW.binfst > outputs/EVW_evaluation.txt")
