import matplotlib.pyplot as plt
from main import fill_template_themed, fill_template
from nltk.corpus import wordnet as wn

STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","at","by","with",
    "is","was","are","were","be","been","being",
    "i","you","he","she","it","we","they","me","him","her","them","my","your",
    "this","that","these","those",
    "as","from","up","down","out","into","about"
} # A basic set of English stopwords

def semantic_similarity(word1, word2): # Computes semantic similarity between two words using WordNet path similarity
    syn1 = wn.synsets(word1)
    syn2 = wn.synsets(word2)
    scores = []

    for s1 in syn1: 
        for s2 in syn2:
            sim = wn.path_similarity(s1, s2)
            if sim is not None:
                scores.append(sim)

    return max(scores) if scores else 0

def run_10_analysis(): # Runs the analysis 10 times and collects semantic similarity scores
    scores = []

    for _ in range(10):
        story_themed = fill_template_themed("The [ADJECTIVE] [NOUN] decided to [VERB] across the [NOUN].")
        story_regular = fill_template("The [ADJECTIVE] [NOUN] decided to [VERB] across the [NOUN].")

        words_t = story_themed.split()
        words_r = story_regular.split()

        for i in range(len(words_t)-1):
            if words_t[i].lower() in STOPWORDS or words_t[i+1].lower() in STOPWORDS:
                continue
            sim = semantic_similarity(words_t[i], words_t[i+1])
            scores.append(sim)

        for i in range(len(words_r)-1):
            if words_r[i].lower() in STOPWORDS or words_r[i+1].lower() in STOPWORDS:
                continue
            sim = semantic_similarity(words_r[i], words_r[i+1])
            scores.append(sim)
            
    return scores

scores = run_10_analysis()

plt.plot(scores, marker='o') # Plotting the semantic similarity scores
plt.title("Semantic Similarity Across Word Pairs (10 runs)")
plt.xlabel("Word Pair Index")
plt.ylabel("Semantic Similarity Score")
plt.show()