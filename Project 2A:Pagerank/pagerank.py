import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    pages = {}

    html_files = [
        f for f in os.listdir(directory)
        if f.endswith(".html")
    ]

    for name in html_files:
        path = os.path.join(directory, name)
        with open(path) as file:
            found = re.findall(
                r'<a\s+(?:[^>]*?)href="([^"]*)"',
                file.read()
            )
            pages[name] = set(found) - {name}

    corpus = set(pages.keys())

    for name in pages:
        pages[name] = {link for link in pages[name] if link in corpus}

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    prop_dist = {}

    # check if page has outgoing links
    dict_len = len(corpus.keys())
    pages_len = len(corpus[page])

    if len(corpus[page]) < 1:
        # no outgoing pages, choosing randomly from all possible pages
        for key in corpus.keys():
            prop_dist[key] = 1 / dict_len

    else:
        # there are outgoing pages, calculating distribution
        random_factor = (1 - damping_factor) / dict_len
        even_factor = damping_factor / pages_len

        for key in corpus.keys():
            if key not in corpus[page]:
                prop_dist[key] = random_factor
            else:
                prop_dist[key] = even_factor + random_factor

    return prop_dist


def sample_pagerank(corpus, damping_factor, n):
    ranks = {page: 0 for page in corpus}
    current = None

    for _ in range(n):
        if current is None:
            current = random.choice(tuple(corpus))
        else:
            probabilities = transition_model(corpus, current, damping_factor)
            pages, weights = zip(*probabilities.items())
            current = random.choices(pages, weights=weights, k=1)[0]

        ranks[current] += 1

    for page in ranks:
        ranks[page] = ranks[page] / n

    return ranks

def iterate_pagerank(corpus, damping_factor):
    n = len(corpus)
    ranks = {page: 1 / n for page in corpus}

    while True:
        updated = {}

        for page in corpus:
            total = 0

            for other in corpus:
                links = corpus[other]
                if links:
                    if page in links:
                        total += ranks[other] / len(links)
                else:
                    total += ranks[other] / n

            updated[page] = (1 - damping_factor) / n + damping_factor * total

        if max(abs(updated[p] - ranks[p]) for p in ranks) < 0.001:
            return updated

        ranks = updated


if __name__ == "__main__":
    main()
