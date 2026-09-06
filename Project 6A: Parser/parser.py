import nltk
import sys
import string

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> PART | PART Conj PART
PART -> NP VP | NP Adv VP | VP
NP -> N | NA N
NA -> Det | Adj | NA NA
VP -> V | V SUPP
SUPP -> NP | P | Adv | SUPP SUPP | SUPP SUPP SUPP
"""
grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    """
    Read a sentence (from file or input),
    preprocess it, parse using grammar,
    then display parse trees and noun phrase chunks.
    """

    # Decide where to get the sentence from
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        with open(filepath, "r") as file:
            sentence = file.read()
    else:
        sentence = input("Sentence: ")

    # Tokenize / preprocess text
    tokens = preprocess(sentence)

    # Try generating parse trees
    try:
        parsed_trees = list(parser.parse(tokens))
    except ValueError as err:
        print(err)
        return

    # If no valid parse
    if len(parsed_trees) == 0:
        print("Could not parse sentence.")
        return

    # Display results
    for parsed_tree in parsed_trees:
        parsed_tree.pretty_print()

        print("Noun Phrase Chunks")
        chunks = np_chunk(parsed_tree)
        for chunk in chunks:
            words = chunk.flatten()
            print(" ".join(words))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    tokenized = nltk.tokenize.word_tokenize(sentence)
    return [x.lower() for x in tokenized if x.isalpha()]

def np_chunk(tree):
    """
    Return only minimal noun phrase chunks.
    A chunk is an NP that does NOT contain any smaller NP inside it.
    """

    chunks = []

    for candidate in tree.subtrees(lambda t: t.label() == "NP"):

        # check if this NP has another NP inside it
        has_inner_np = any(
            child.label() == "NP"
            for child in candidate.subtrees()
            if child is not candidate
        )

        if not has_inner_np:
            chunks.append(candidate)

    return chunks


if __name__ == "__main__":
    main()
