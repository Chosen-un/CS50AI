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
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    probabilities = {}

    linked_pages = corpus[page]
    num_links = len(linked_pages)

    if num_links == 0:
        for p in corpus:
            probabilities[p] = 1 / num_pages
        return probabilities

    random_prob = (1 - damping_factor) / num_pages

    linked_prob = damping_factor / num_links

    for p in corpus:
        probabilities[p] = random_prob
        if p in linked_pages:
            probabilities[p] += linked_prob
            
    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_visits = {page: 0 for page in corpus}
    
    current_sample = random.choice(list(corpus.keys()))
    page_visits[current_sample] += 1

    for _ in range(n - 1):
        model = transition_model(corpus, current_sample, damping_factor)
        pages = list(model.keys())
        weights = list(model.values())
        current_sample = random.choices(pages, weights, k=1)[0]
        page_visits[current_sample] += 1
    
    pageranks = {page: count / n for page, count in page_visits.items()}
    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    pageranks = {page: 1 / num_pages for page in corpus}
    
    links_to_page = {p: set() for p in corpus}
    for page, links in corpus.items():
        if not links:
            for p in corpus:
                links_to_page[p].add(page)
        else:
            for link in links:
                links_to_page[link].add(page)

    while True:
        max_change = 0
        new_pageranks = {}

        for page in corpus:
            first_term = (1 - damping_factor) / num_pages
            
            summation = 0
            for linking_page in links_to_page[page]:
                num_links = len(corpus[linking_page])
                if num_links == 0:
                    num_links = num_pages
                summation += pageranks[linking_page] / num_links
            
            second_term = damping_factor * summation
            
            new_rank = first_term + second_term
            new_pageranks[page] = new_rank
            
            change = abs(pageranks[page] - new_rank)
            if change > max_change:
                max_change = change

        pageranks = new_pageranks

        if max_change < 0.001:
            break
    
    return pageranks

if __name__ == "__main__":
    main()
