#!/usr/bin/env zsh

# Run spell corrector for an input word

# Usage:
#   bash scripts/predict.sh MY_SPELL_CHECKER tst
# Output:
#   test

# Command line args
SPELL_CHECKER_COMPILED=${1}
WORD=${2}


# Constants.
CURRENT_DIRECTORY=$(dirname ${0})

###
# Make sure these files exist
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms
WORDSYMS=${CURRENT_DIRECTORY}/../vocab/words40.syms
###

# Make input fst for the misspelled word
python scripts/mkfstinput.py ${WORD} |
    # Compile and compose with the spell checker
    fstcompile --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} |
    fstcompose - ${SPELL_CHECKER_COMPILED} |
    # Get shortest path and sort arcs
    fstshortestpath |
    fstrmepsilon |
    fsttopsort |
    # print output fst using words.syms
    fstprint -osymbols=${WORDSYMS} |
    # Get destination word (corrected)
    cut -f4 |
    # Ignore epsilon outputs
    grep -v "<eps>" |
    # Ignore accepting state line
    sed '$ d' |         #head was not working
    # Remove trailing new line
    tr -d '\n'
