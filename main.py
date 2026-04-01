"""
AI Mad Libs Project
Main program script for generating stories using NLP-based word selection.
"""
# =============================================================================
# AI-Powered Mad Libs Generator
# This script takes story templates and intelligently fills in blanks using:
# - part-of-speech tagging
# - semantic and contextual scoring
# - humor scoring to keep stories engaging and varied
# - theme-based vocabulary
# The goal of this project is to create stories that are both grammatically
# correct and entertaining, while also demonstrating real NLP logic in action.
# =============================================================================

# ========================
# IMPORTS
# ========================
# Standard library stuff - random for randomness, re for regex, math for cosine similarity
import random
import re
import math
# NLTK's WordNet is basically our massive dictionary/thesaurus
from nltk.corpus import wordnet as wn
# Our custom templates live in templates.py
from templates import TEMPLATES
# Spacy does the heavy NLP work (POS tagging, lemmatization, etc.)
import spacy
# Our themed word lists for different vibes (spooky, sci-fi, adventure, etc.)
from themes import THEMED_WORDS

# Load spacy's English model - this is what gives us all the NLP superpowers
nlp = spacy.load("en_core_web_sm")


# ========================
# CURATED WORD LISTS
# ========================
# WordNet is awesome but it's also got like 10,000 words nobody's heard of.
# These curated lists are hand-picked words that are:
# 1. Actually recognizable
# 2. Work well in stories
# 3. Ensure the generated story remains natural and readable
# Think of these as our "approved vocabulary" for safer word choices.

CURATED_NOUNS = [
    
    # People - your standard cast of characters
    'person', 'man', 'woman', 'child', 'baby', 'friend', 'stranger', 'teacher', 
    'doctor', 'student', 'worker', 'player', 'artist', 'musician',
    
    # Animals - from pets to mythical beasts
    'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig', 'lion', 'bear', 'wolf',
    'dragon', 'monster', 'creature', 'animal',
    
    # Objects - Common stuff you'd actually find in real life
    'book', 'phone', 'computer', 'car', 'bike', 'ball', 'toy', 'box', 'bag',
    'cup', 'plate', 'bottle', 'pen', 'paper', 'key', 'door', 'window', 'chair', 'table',
    
    # Objects - Fantasy/Adventure gear for when things get epic
    'sword', 'shield', 'crown', 'ring', 'gem', 'crystal', 'scroll', 'map', 'treasure',
    'potion', 'spell', 'wand', 'staff', 'amulet',
    
    # Places - settings for our stories
    'house', 'room', 'city', 'town', 'forest', 'mountain', 'river', 'ocean', 'beach',
    'school', 'store', 'park', 'road', 'path',
    
    # Food - because every good story needs snacks
    'pizza', 'burger', 'sandwich', 'cake', 'cookie', 'apple', 'bread', 'cheese',
    'coffee', 'water', 'juice', 'meal', 'snack',
    
    # Abstract/Concepts - the intangible stuff that makes stories interesting
    'idea', 'plan', 'secret', 'answer', 'question', 'story', 'song', 'game',
    'power', 'magic', 'luck', 'chance', 'hope', 'dream', 'wish', 'gift', 'prize',
    
    # Fantasy beings - for when you need something supernatural
    'ghost', 'spirit', 'wizard', 'witch', 'knight', 'warrior', 'guardian',
    'demon', 'angel', 'fairy', 'giant', 'alien', 'robot'
]

# These words are objectively hilarious and get bonus points in our humor scoring.
# Humor scoring is subjective and based on general tendencies.
FUNNY_NOUNS = {'banana', 'platypus', 'cucumber', 'noodle', 'bucket', 'squirrel', 'donkey',
               'waffle', 'taco', 'meatball', 'unicorn', 'goblin', 'ferret', 'hamster', 'pickle'}

CURATED_VERBS = [
    # Basic movement - getting from A to B
    'walk', 'run', 'jump', 'climb', 'fly', 'swim', 'crawl', 'slide', 'dance',
    'move', 'go', 'come', 'leave', 'arrive', 'enter', 'exit', 'fall', 'rise',
    
    # Physical actions - transitive (these need objects to work with)
    'touch', 'grab', 'hold', 'carry', 'throw', 'catch', 'push', 'pull', 'lift',
    'drop', 'break', 'fix', 'open', 'close', 'turn', 'press', 'shake',
    'eat', 'drink', 'cook', 'bake', 'wash', 'clean', 'build', 'make', 'create',
    'buy', 'sell', 'give', 'take', 'find', 'lose', 'hide', 'show',
    
    # Communication
    'say', 'tell', 'ask', 'talk', 'speak', 'whisper', 'shout', 'yell', 'call',
    'sing', 'laugh', 'cry', 'scream', 'smile',
    
    # Perception - experiencing the world through senses
    'see', 'look', 'watch', 'stare', 'gaze', 'hear', 'listen', 'smell', 'taste', 'feel',
    
    # Mental - brain stuff
    'think', 'know', 'believe', 'remember', 'forget', 'learn', 'understand',
    'wonder', 'imagine', 'dream', 'hope', 'wish', 'want', 'need', 'like', 'love', 'hate',
    
    # Continuous/ongoing - perfect for "started VERBing" constructions
    'glow', 'shine', 'shimmer', 'sparkle', 'flicker', 'pulse', 'vibrate', 'hum',
    'spin', 'rotate', 'grow', 'shrink', 'change', 'transform', 'fade', 'disappear',
    'melt', 'freeze', 'burn', 'float', 'sink', 'tremble', 'shake',
    
    # Intentional - good for "would VERB" scenarios
    'help', 'save', 'protect', 'guard', 'defend', 'attack', 'fight', 'destroy',
    'serve', 'follow', 'lead', 'obey', 'trust', 'join', 'leave', 'stay',
    'try', 'attempt', 'start', 'stop', 'continue', 'finish', 'win', 'lose',
    
    # Work/Activity
    'work', 'play', 'study', 'read', 'write', 'draw', 'paint', 'practice'
]

CURATED_ADJECTIVES = [
    # Size - describing scale
    'big', 'large', 'huge', 'giant', 'enormous', 'small', 'tiny', 'little', 'miniature',
    
    # Age - how old something is
    'old', 'ancient', 'new', 'young', 'modern', 'fresh',
    
    # Quality - Positive (the good stuff)
    'good', 'great', 'wonderful', 'amazing', 'perfect', 'beautiful', 'lovely',
    'nice', 'pretty', 'cute', 'cool', 'awesome', 'fantastic',
    
    # Quality - Negative (the bad stuff we also need)
    'bad', 'terrible', 'awful', 'horrible', 'ugly', 'nasty', 'gross', 'weird',
    'strange', 'odd', 'bizarre', 'creepy', 'scary', 'spooky',
    
    # Emotion - feelings and vibes
    'happy', 'sad', 'angry', 'mad', 'excited', 'bored', 'tired', 'sleepy',
    'scared', 'afraid', 'brave', 'proud', 'lonely', 'jealous',
    
    # Color - ROY G. BIV and friends
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white',
    'brown', 'gray', 'golden', 'silver',
    
    # Temperature - hot or cold (literally)
    'hot', 'warm', 'cold', 'cool', 'freezing', 'boiling', 'icy',
    
    # Speed - velocity descriptors
    'fast', 'quick', 'rapid', 'slow', 'lazy',
    
    # Sound/Light - sensory descriptions
    'loud', 'noisy', 'quiet', 'silent', 'soft', 'bright', 'dark', 'dim', 'shiny', 'dull',
    
    # Texture/Physical - how stuff feels
    'hard', 'soft', 'smooth', 'rough', 'sharp', 'dull', 'wet', 'dry', 'clean', 'dirty',
    'heavy', 'light',
    
    # Mystery/Magic - for fantasy contexts
    'magical', 'mysterious', 'mystical', 'enchanted', 'haunted', 'cursed', 'blessed',
    'sacred', 'secret', 'hidden', 'forbidden', 'lost', 'forgotten',
    
    # Appearance - visual qualities
    'glowing', 'shimmering', 'sparkling', 'faint', 'vivid', 'pale', 'shadowy', 'ghostly'
]
# ========================
# QUALITY FILTER (prevents archaic / weird words)
# ========================
def word_quality_penalty(word):
    # avoid weird academic/archaic words
    if len(word) > 10:
        return -2.0
    if word.count(" ") > 0:
        return -1.0
    # no non-ASCII stuff (foreign, archaic, obscure)
    if not word.isascii():
        return -3.0
    # reject archaic-looking endings
    if word.endswith(("eth", "est", "ith", "th")):
        return -3.0
    # reject super uncommon / old-sounding forms
    if word in {"hitherto","thereof","wherein","whereof","thusly","aught","nigh"}:
        return -5.0
    return 0.0

# ========================
# WORDNET WORD EXTRACTION
# ========================
def get_words_for_pos(pos, min_frequency=3):
    """
    Get commonly used words from WordNet for a specific part of speech.

    This function:
    - pulls all lemmas for the POS
    - keeps only the more frequent words
    - removes words that are too long or unusual
    - validates POS using spaCy

    The goal is to build a useful, readable vocabulary for story generation.
    """
    word_count = {}
    
    # Loop through all synonym sets in WordNet for this part of speech
    for syn in wn.all_synsets(pos):
        for lemma in syn.lemmas():
            # Clean up the word - remove underscores, make it lowercase
            word = lemma.name().replace("_", " ").lower()
            # Filter: must be alphabetic, reasonable length, single words only
            if word.isalpha() and len(word) <= 10 and word.count(" ") == 0:
                # Track how common this word is using lemma count
                word_count[word] = word_count.get(word, 0) + lemma.count()
    
    # Sort by frequency and keep top 30% - these are our most useful words
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    cutoff = len(sorted_words) // 3
    
    # Double-check with spacy that these are actually the right POS
    # (WordNet sometimes gets creative with classifications)
    validated = []
    expected_pos = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 's': 'ADJ'}[pos]
    
    for word, count in sorted_words[:cutoff] if cutoff > 0 else sorted_words:
        doc = nlp(word)
        # Only include if spacy confirms this is the right part of speech
        if doc and doc[0].pos_ == expected_pos:
            validated.append(word)
    
    # Return validated list, or fall back to unvalidated if validation failed
    return validated if validated else [w for w, _ in (sorted_words[:cutoff] if cutoff > 0 else sorted_words)]

# Curated-first vocabulary with WordNet fallback
# Curated lists are the primary source; WordNet items are appended only if not duplicates.

WN_NOUNS = get_words_for_pos('n')
WN_VERBS = [w for w in get_words_for_pos('v') if nlp(w)[0].pos_ == "VERB"]
WN_ADJECTIVES = get_words_for_pos('a')

# Merge curated lists with WordNet fallback (curated words first)
NOUNS = CURATED_NOUNS + [w for w in WN_NOUNS if w not in CURATED_NOUNS]
VERBS = CURATED_VERBS + [w for w in WN_VERBS if w not in CURATED_VERBS]
ADJECTIVES = CURATED_ADJECTIVES + [w for w in WN_ADJECTIVES if w not in CURATED_ADJECTIVES]

# ========================
# SIMILARITY CALCULATIONS
# ========================
def word_to_vector(word):
    """
    Convert a word into a basic letter-frequency vector.

    Each vector has 26 numbers (one per letter).  
    This gives us a simple numerical representation we can use for similarity.
    """
    vec = [0] * 26  # One slot for each letter a-z
    for c in word.lower():
        if c.isalpha():
            # Map a=0, b=1, c=2, ... z=25
            vec[ord(c) - 97] += 1
    return vec

def cosine_sim(v1, v2):
    """
    Compute cosine similarity between two vectors.

    Returns a value between 0 and 1  
    that measures how similar the vectors are in direction.
    """
    dot = sum(a*b for a, b in zip(v1, v2))  # Dot product
    mag1 = math.sqrt(sum(a*a for a in v1))   # Magnitude of v1
    mag2 = math.sqrt(sum(a*a for a in v2))   # Magnitude of v2

    # Avoid division by zero because that breaks everything
    if mag1 == 0 or mag2 == 0:
        return 0

    return dot / (mag1 * mag2)

# ========================
# SEMANTIC SIMILARITY
# ========================
def get_semantic_similarity(word, context_words):
    """
    Estimate how related a word is to nearby context words using WordNet.

    Two methods are combined:
    1. path similarity between synsets
    2. definition word overlap

    Higher scores mean the word fits the context better.
    """
    if not context_words:
        return 0  # No context = can't calculate similarity
    
    # Get all the meanings (synsets) for our candidate word
    word_synsets = wn.synsets(word)
    if not word_synsets:
        return 0  # Word not in WordNet? We're out of luck.
    
    # Method 1: Path similarity
    # Checks how words are related in WordNet's semantic network
    path_scores = []
    for context_word in context_words[-5:]:  # Only look at last 5 words for efficiency
        context_synsets = wn.synsets(context_word)
        for w_syn in word_synsets:
            for c_syn in context_synsets:
                try:
                    sim = w_syn.path_similarity(c_syn)
                    if sim:
                        path_scores.append(sim)
                except:
                    pass  # Some pairs don't have paths, that's fine
    
    # Method 2: Definition overlap
    # Get the dictionary definitions and see how much they overlap
    word_definitions = " ".join([syn.definition() for syn in word_synsets[:3]])
    word_def_words = set(word_definitions.lower().split())
    
    context_definitions = []
    for cw in context_words[-5:]:
        for syn in wn.synsets(cw)[:2]:
            context_definitions.append(syn.definition())
    
    context_def_words = set(" ".join(context_definitions).lower().split())
    
    # Calculate overlap - how many words do the definitions share?
    overlap = len(word_def_words & context_def_words)
    overlap_score = overlap / 20.0  # Normalize to 0-1 range
    
    # Combine both methods (weighted average)
    path_score = sum(path_scores) / len(path_scores) if path_scores else 0
    
    # Path similarity (60%) + definition overlap (40%)
    return (path_score * 0.6) + (overlap_score * 0.4)

# ========================
# HUMOR SCORING
# ========================
def humor_score(candidate_word, context_words):
    """
    Give a small “humor bonus” to words that are surprising or commonly funny.

    Looks at:
    - intentionally odd word pairs
    - a small list of funny words
    - word length
    - low semantic similarity (more unexpected = funnier)

    Used to add variety to the generated story.
    """
    if not context_words:
        return 0.0

    score = 0.0

    # Selected word combinations that are intentionally surprising or humorous    
    absurd_pairs = [
        ("wizard", "laptop"), ("ghost", "pizza"), ("dragon", "taxes"),
        ("unicorn", "homework"), ("alien", "football"), ("monster", "therapy")
    ]
    for cw in context_words[-5:]:
        if (candidate_word, cw) in absurd_pairs or (cw, candidate_word) in absurd_pairs:
            score += 1.2  # Big boost for absurd pairs

    # Some words consistently add mild humor to the story
    funny_words = {"banana", "fart", "booger", "pickle", "noodle", "clown", "poop", "butt"}
    # Reduce humor bonus unless humorous context exists
    if candidate_word in funny_words:
        # Only give full humor bonus if story is not serious so far
        if not any(w in context_words[-10:] for w in ['escaped', 'danger', 'monster', 'battle']):
            score += 0.4   # lower boost, not +1.0

    # Shorter words can create more impact and variety
    if len(candidate_word) <= 4 and candidate_word not in ["and", "the", "with", "from"]:
        score += 0.2

    # Semantic distance = unexpectedness = comedy
    # Words that don't quite fit are often the funniest
    if context_words:
        semantic_fit = get_semantic_similarity(candidate_word, context_words)
        score += (0.4 - semantic_fit)  # Less fit = more funny

    return max(score, 0.0)  # Keep score non-negative

# ========================
# VERB CONJUGATION
# ========================
def detect_verb_context(text_before_verb):
    """
    Try to determine what verb form is needed based on the previous few words.

    Checks for patterns like:
    - modals (“will”, “could”) → base form
    - “was/were/is/are” → gerund
    - prepositions → gerund
    - past-tense context → past

    Returns the tense to use and whether it should be progressive.
    """
    # Clean up the text
    text = text_before_verb.strip().lower()
    
    if not text:
        return "BASE", False  # No context? Use base form.
    
    # Split into words for easier analysis
    words = text.split()
    if not words:
        return "BASE", False
    
    last_word = words[-1]
    last_2_words = " ".join(words[-2:]) if len(words) >= 2 else last_word
        
    # After prepositions → use GERUND
    # Examples: "by running", "without talking", "after eating"
    if last_word in ["by", "without", "after", "before", "for", "about", "of", "from", "in", "on"]:
        return "GERUND", False
    
    # After these verbs → use GERUND
    # Examples: "started running", "kept talking", "stopped eating"
    if last_word in ["started", "stopped", "kept", "continued", "finished", "began"]:
        return "GERUND", False
    
    # After "be" verbs → usually GERUND
    # Examples: "was running", "is talking", "were eating"
    if last_word in ["was", "were", "is", "are", "am", "been", "be"]:
        return "GERUND", False
        
    # After modal verbs → use BASE form
    # Examples: "will run", "could talk", "should go"
    if last_word in ["will", "would", "could", "should", "can", "may", "might", "must", "shall"]:
        return "BASE", False
    
    # After "to" → use BASE form
    # Examples: "to run", "to talk", "to go"
    if last_word == "to":
        return "BASE", False
    
    # After "did", "doesn't", etc. → use BASE form
    # Examples: "didn't run", "doesn't talk", "does go"
    if last_word in ["did", "didn't", "doesn't", "don't", "does"]:
        return "BASE", False
        
    # After pronouns in past context → might need PAST
    if last_word == "i" and any(w in words for w in ["found", "was", "felt", "wasn't"]):
        return "PAST", False
    
    if last_word == "we" and any(w in words for w in ["found", "was", "were", "felt"]):
        return "PAST", False
    
    # After "and" → check what verb came before for parallelism
    # Example: "I walked and ___" should be "ran" (past tense to match)
    if last_word == "and":
        doc = nlp(text_before_verb)
        for i in range(len(doc) - 2, max(-1, len(doc) - 8), -1):
            if i >= 0 and doc[i].pos_ == "VERB":
                if doc[i].tag_ == "VBD":  # Past tense
                    return "PAST", False
                elif doc[i].tag_ == "VBG":  # Gerund
                    return "GERUND", False
                break
    
    # Check if the overall context suggests past tense
    doc = nlp(text_before_verb)
    recent_tokens = list(doc)[-8:] if len(doc) > 8 else list(doc)
    
    for token in recent_tokens:
        if token.tag_ == "VBD":  # Found a past tense verb nearby
            return "PAST", False
    
    # Default to present tense
    return "PRESENT", False

def conjugate_verb(verb, tense, progressive=False):
    """
    Conjugate a verb into the needed form.

    Attempts spaCy inflection first,  
    then uses simple rule-based fallbacks for:
    - base form
    - gerund
    - past tense
    - present tense
    - future (base form after “will”)
    """
    token = nlp(verb)[0]
    
    try:
        # Irregular verbs map
        irregular_past = {
            'draw': 'drew',
            'blow': 'blew',
            'throw': 'threw',
            'grow': 'grew',
            'know': 'knew',
            'fly': 'flew',
            'run': 'ran',
            'begin': 'began',
            'swim': 'swam',
            'sing': 'sang',
            'ring': 'rang',
            'write': 'wrote',
            'ride': 'rode',
            'speak': 'spoke',
            'break': 'broke'
        }
        if tense == "PAST" and verb in irregular_past:
            return irregular_past[verb]
        # Try spaCy's inflection first (if available)
        if tense == "BASE":
            base = token._.inflect("VB")
            return base if base else verb
        
        if tense == "GERUND" or progressive:
            ing = token._.inflect("VBG")
            return ing if ing else verb + "ing"

        if tense == "PAST":
            past = token._.inflect("VBD")
            return past if past else verb

        elif tense == "FUTURE":
            return verb  # Just use base form after "will"

        else:  # PRESENT (3rd person singular)
            pres = token._.inflect("VBZ")
            return pres if pres else verb
        
    except AttributeError:
        # Fallback: Use simple rule-based conjugation when spaCy doesn't have inflection
        
        if tense == "BASE":
            return verb
        
        elif tense == "GERUND" or progressive:
            # Rules for adding -ing
            if verb.endswith('e') and not verb.endswith('ee'):
                return verb[:-1] + "ing"  # hope → hoping
            elif verb.endswith('ie'):
                return verb[:-2] + "ying"  # die → dying
            # Double consonant for CVC pattern (consonant-vowel-consonant)
            elif len(verb) >= 3 and verb[-1] not in 'aeiou' and verb[-2] in 'aeiou' and verb[-3] not in 'aeiou':
                return verb + verb[-1] + "ing"  # run → running
            else:
                return verb + "ing"
        
        elif tense == "PAST":
            # Rules for past tense
            if verb.endswith('e'):
                if verb.endswith('ow'):
                    return verb + "ed"  # show → showed
                return verb + "d"  # hope → hoped
            elif verb.endswith('y') and len(verb) > 2 and verb[-2] not in 'aeiou':
                return verb[:-1] + "ied"  # cry → cried
            # Double consonant for CVC pattern
            elif len(verb) >= 3 and verb[-1] not in 'aeiou' and verb[-2] in 'aeiou' and verb[-3] not in 'aeiou':
                return verb + verb[-1] + "ed"  # stop → stopped
            else:
                return verb + "ed"
        
        elif tense == "FUTURE":
            return verb
        
        else:  # PRESENT (3rd person singular)
            if verb.endswith(('s', 'x', 'z', 'ch', 'sh')):
                return verb + "es"  # wash → washes
            elif verb.endswith('y') and len(verb) > 2 and verb[-2] not in 'aeiou':
                return verb[:-1] + "ies"  # fly → flies
            else:
                return verb + "s"  # walk → walks

# ========================
# CONTEXTUAL APPROPRIATENESS
# ========================
def is_contextually_appropriate(word, text_before, pos_type):
    """
    Check whether a proposed word fits the local context.

    Uses basic rules based on surrounding words,  
    avoiding combinations that sound confusing or unnatural  
    (e.g., intransitive verbs after objects, abstract nouns after certain verbs).
    """
    text_lower = text_before.lower()
    words = text_lower.split()
    
    if len(words) < 2:
        return True  # Not enough context to judge
    
    last_word = words[-1]
    last_3 = words[-3:] if len(words) >= 3 else words
    
    # Rules for VERBS
    if pos_type == 'VERB':
        # After modal verbs → avoid stative/state-of-being verbs
        # "I would run" is good; "I would seem" is weird
        if last_word in ["would", "could", "should", "might", "may"]:
            avoid = ['seem', 'appear', 'become', 'feel', 'look', 'sound', 'remain']
            return word not in avoid
        
        # After "started", "began" → avoid instantaneous/punctual verbs
        # "started running" works; "started finding" is weird
        if last_word in ["started", "began", "kept", "continued", "stopped", "finished"]:
            avoid = ['find', 'lose', 'discover', 'arrive', 'leave', 'win', 'die', 'know', 'realize']
            return word not in avoid
        
        # After pronouns/objects → avoid intransitive verbs
        # "grabbed it" works; "arrived it" doesn't
        if last_word in ["it", "them", "him", "her", "this", "that"]:
            avoid = ['arrive', 'exist', 'occur', 'happen', 'sleep', 'die', 'live']
            return word not in avoid
        
        # After "to" → filter out nonsensical base forms
        if last_word == "to":
            avoid = ['is', 'are', 'was', 'were', 'been', 'being']
            return word not in avoid
    
    # Rules for NOUNS
    if pos_type == 'NOUN':
        # Avoid abstract nouns after verbs like packed / carried / brought
        packing_verbs = ['packed', 'carried', 'brought', 'grabbed', 'took']
        if any(v in words[-3:] for v in packing_verbs):
            abstract_nouns = [
                'hope', 'dream', 'idea', 'plan', 'chance', 'luck', 'power', 'magic',
                'secret', 'answer', 'question'
            ]
            return word not in abstract_nouns
        # After verbs suggesting appearance → prefer animate/agent nouns
        # "A creature appeared" is good; "A theory appeared" is weird
        if any(v in last_3 for v in ['appeared', 'emerged', 'arrived', 'came', 'entered', 'walked']):
            animate = ['person', 'creature', 'animal', 'being', 'figure', 'stranger', 
                      'friend', 'enemy', 'monster', 'ghost', 'spirit', 'warrior',
                      'wizard', 'knight', 'dragon', 'alien', 'robot', 'child', 'man', 'woman']
            if word in animate:
                return True
            # Allow other nouns but with lower probability
            return random.random() < 0.4
        
        # After possession verbs → prefer concrete/valuable nouns
        # "promised me a gift" works; "promised me an absence" doesn't
        if any(v in last_3 for v in ['promised', 'gave', 'offered', 'showed', 'brought', 'handed']):
            abstract_bad = ['absence', 'lack', 'nothing', 'nobody', 'nowhere']
            return word not in abstract_bad
        
        # After "a"/"an" + adjective → avoid overly abstract concepts
        if len(last_3) >= 2 and last_3[-2] in ['a', 'an']:
            avoid = ['theory', 'concept', 'idea', 'thought', 'notion']
            return word not in avoid
    
    # Rules for ADJECTIVES
    if pos_type == 'ADJECTIVE':
        # Avoid emotional or character adjectives after physical nouns like "path"
        if any(w in text_before.lower().split()[-3:] for w in ['path', 'road', 'trail', 'way']):
            bad_adj = [
                'sober', 'jealous', 'angry', 'proud', 'lonely', 'happy',
                'sad', 'weird', 'bored'
            ]
            return word not in bad_adj
        # Adjectives are pretty flexible, but don't use nouns as adjectives
        avoid_noun_adjectives = ['person', 'thing', 'place', 'time']
        return word not in avoid_noun_adjectives
    
    return True  # Default: allow the word

# ========================
# WORD SELECTION (NON-THEMED)
# ========================
def choose_ai_word(pos_list, context_words, pos_type=None, randomness=0.3, text_before=""):
    """
    Choose a word that matches the desired part of speech
    and fits the surrounding context.

    Steps:
    - prefer curated lists
    - filter out contextually odd choices
    - score remaining options using semantic fit + humor
    - pick with weighted randomness

    Used for non-themed word selection.
    """
    
    # Step 1: Figure out which curated list to use
    curated_list = None
    if pos_type == 'NOUN':
        curated_list = CURATED_NOUNS
    elif pos_type == 'VERB':
        # Force movement verbs in motion contexts
        motion_triggers = ['forward', 'ahead', 'quickly', 'fast', 'running']
        if any(w in text_before.lower().split()[-3:] for w in motion_triggers):
            movement_verbs = ['run', 'walk', 'dash', 'race', 'sprint', 'move', 'hurry', 'charge']
            filtered = [v for v in movement_verbs if v in pos_list]
            if filtered:
                return random.choice(filtered)
        curated_list = CURATED_VERBS
    elif pos_type == 'ADJECTIVE':
        curated_list = CURATED_ADJECTIVES
    
    # Step 2: Filter curated list for context appropriateness
    if curated_list:
        appropriate = [w for w in curated_list 
                      if w in pos_list and is_contextually_appropriate(w, text_before, pos_type)]
        
        if appropriate:
            # 70% chance to use curated words (they're more reliable)
            if random.random() < 0.7:
                if not context_words:
                    return random.choice(appropriate)  # No context? Pick randomly.
                
                # Score curated words by semantic similarity
                scored = []
                for w in appropriate:
                    score = get_semantic_similarity(w, context_words)
                    scored.append((w, score))
                
                scored.sort(key=lambda x: x[1], reverse=True)

                # Take top half
                cutoff = max(5, len(scored) // 2)  
                top = scored[:cutoff]
                
                # Weighted random choice - higher scores more likely
                weights = [(score + 0.1) for _, score in top]
                return random.choices([w for w, _ in top], weights=weights)[0]
    
    # Step 3: Fallback to full WordNet vocabulary (more diverse but riskier)
    if not context_words:
        simple_words = [w for w in pos_list if 3 <= len(w) <= 10 and w.isalpha()]
        return random.choice(simple_words) if simple_words else random.choice(pos_list)
    
    # Step 4: Score all remaining candidates using semantic and humor metrics
    scored = []
    for w in pos_list:
        # Filter out weird/long words
        if len(w) > 12 or "-" in w or not w.isalpha():  
            continue
        
        # Be extra careful with abstract noun suffixes
        if any(w.endswith(suffix) for suffix in ['tion', 'ism']):
            if len(w) > 10:  
                continue
        
        # Check contextual appropriateness
        if not is_contextually_appropriate(w, text_before, pos_type):
            continue
        
        # Calculate composite score: semantic fit + humor
        semantic = get_semantic_similarity(w, context_words)
        humor = humor_score(w, context_words)
        score = semantic + humor
        
        # Bonus for inherently funny nouns
        if pos_type == 'NOUN' and w in FUNNY_NOUNS:
            score += 1.2
        
        # Bonus for animate nouns when context suggests something is acting
        # "A wizard appeared" > "A theory appeared"
        if pos_type == 'NOUN':
            animate = ['person', 'creature', 'animal', 'being', 'figure', 'stranger', 
                      'friend', 'enemy', 'monster', 'ghost', 'spirit', 'warrior',
                      'wizard', 'knight', 'dragon', 'alien', 'robot', 'child', 'man', 'woman']
            if w in animate and any(v in text_before.lower() for v in ['appeared', 'started', 'began', 'walked', 'arrived']):
                score += 1.5
        
        scored.append((w, score))
    
    if not scored:
        # Emergency fallback - just pick something that's alphabetic
        return random.choice([w for w in pos_list if w.isalpha()][:100])
    
    # Sort by score and take top 30%
    scored.sort(key=lambda x: x[1], reverse=True)
    cutoff = max(10, int(len(scored) * 0.3))  
    top_candidates = scored[:cutoff]
    
    # Weighted random choice - better scores are more likely but not guaranteed
    weights = [(score + 0.1) ** 1.2 for _, score in top_candidates]  
    return random.choices([w for w, _ in top_candidates], weights=weights)[0]

# ========================
# THEME SELECTION
# ========================
def choose_theme():
    """
    Display available themes and let the user pick one.

    If the user selects “random”,  
    a theme is chosen automatically.
    """
    print("\n=== SELECT A THEME ===")
    themes = list(THEMED_WORDS.keys())
    themes.append('random')  # Allows a randomly selected theme
    
    for i, theme in enumerate(themes, 1):
        print(f"{i}. {theme.capitalize()}")
    
    while True:
        try:
            choice = input(f"\nEnter theme number (1-{len(themes)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(themes):
                selected = themes[idx]
                print(f"\n✨ Theme selected: {selected.upper()} ✨\n")
                return selected if selected != 'random' else None
            else:
                print(f"Please enter a number between 1 and {len(themes)}")
        except ValueError:
            print("Please enter a valid number")

def get_themed_word_list(pos_type, theme=None):
    """
    Return the appropriate word list for a specific part of speech.

    If a theme is active, use themed vocabulary.  
    Otherwise fall back to general curated lists.
    """
    if theme and theme in THEMED_WORDS:
        themed_list = THEMED_WORDS[theme].get(pos_type, [])
        if themed_list:
            return themed_list
    
    # Fallback to curated lists
    if pos_type == 'NOUN':
        return CURATED_NOUNS
    elif pos_type == 'VERB':
        return CURATED_VERBS
    elif pos_type == 'ADJECTIVE':
        return CURATED_ADJECTIVES
    return []

# ========================
# WORD SELECTION (THEMED)
# ========================
def choose_ai_word_themed(pos_list, context_words, pos_type=None, randomness=0.3, text_before="", theme=None):
    """
    Select a word like choose_ai_word(), but with theme awareness.

    Themed words are preferred when available.  
    Still uses contextual filtering, semantic scoring, and humor weighting.
    """
    
    # Step 1: Get themed vocabulary
    themed_list = get_themed_word_list(pos_type, theme)
    
    # Step 2: Filter for context + existence in WordNet
    appropriate = []
    for w in themed_list:
        # Make sure word actually exists in WordNet for this POS
        if w not in pos_list:
            continue
        
        # Check if it makes sense in this context
        if not is_contextually_appropriate(w, text_before, pos_type):
            continue
        
        # Extra filtering for verbs - avoid abstract verbs in action contexts
        if pos_type == 'VERB':
            # If we're talking about movement/action, skip abstract verbs
            if any(word in text_before.lower() for word in ['forward', 'away', 'fast', 'quickly', 'escaped']):
                abstract_verbs = ['curse', 'rot', 'decay', 'haunt', 'possess']
                if w in abstract_verbs:
                    continue
        
        # Extra filtering for adjectives - prefer sensory in physical contexts
        if pos_type == 'ADJECTIVE':
            # For physical descriptions, avoid overly abstract adjectives
            if any(word in text_before.lower() for word in ['path', 'road', 'way', 'trail']):
                abstract_adj = ['cultural', 'spiritual', 'emotional']
                if w in abstract_adj:
                    continue
        
        appropriate.append(w)
    
    # Step 3: Use themed words with HIGH probability
    if appropriate:
        # 85% chance for themed words (we REALLY want that theme to shine through)
        theme_weight = 0.85 if theme else 0.7
        if random.random() < theme_weight:
            if not context_words:
                return random.choice(appropriate)  # No context? Random themed word.
            
            # Score themed words
            scored = []
            for w in appropriate:
                semantic = get_semantic_similarity(w, context_words)
                humor = humor_score(w, context_words)
                score = semantic + humor
                score += word_quality_penalty(w) # Penalize low-quality words
                
                # Bonus for funny nouns
                if pos_type == 'NOUN' and w in FUNNY_NOUNS:
                    score += 1.2
                
                # Bonus for animate nouns in appropriate contexts
                if pos_type == 'NOUN':
                    animate = ['person', 'creature', 'animal', 'being', 'figure', 'stranger', 
                              'friend', 'enemy', 'monster', 'ghost', 'spirit', 'warrior',
                              'wizard', 'knight', 'dragon', 'alien', 'robot', 'child', 'man', 'woman']
                    if w in animate and any(v in text_before.lower() for v in ['appeared', 'started', 'began', 'walked', 'arrived']):
                        score += 1.5
                
                scored.append((w, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Take top third (minimum 3 options)
            cutoff = max(3, len(scored) // 3)
            top = scored[:cutoff]
            
            # Weighted random choice
            weights = [(score + 0.2) for _, score in top]
            return random.choices([w for w, _ in top], weights=weights)[0]
    
    # Step 4: Fallback to non-themed logic
    return choose_ai_word(pos_list, context_words, pos_type, randomness, text_before)

# ========================
# TEMPLATE FILLING (THEMED)
# ========================
def fill_template_themed(template, theme=None):
    """
    Fill a Mad Libs template, optionally using a theme.

    Handles:
    - extracting placeholders
    - selecting words with context and theme logic
    - verb tense detection and conjugation
    - avoiding repeated words

    Returns a completed story.
    """
    placeholders = re.findall(r"\[(.*?)\]", template)
    filled = template
    context = []  # Track words for semantic similarity
    used_words = set()  # Avoid repetition
    main_subject = None  # Track recurring character/object

    for ph in placeholders:
        # Get everything before this placeholder for context
        text_before = filled.split(f"[{ph}]")[0]
        
        if ph == "NOUN":
            replacement = choose_ai_word_themed(NOUNS, context, pos_type='NOUN', 
                                               text_before=text_before, theme=theme)
            
            # Try to avoid reusing words (up to 15 attempts)
            attempts = 0
            while replacement in used_words and attempts < 40:
                replacement = choose_ai_word_themed(NOUNS, context, pos_type='NOUN', 
                                                   text_before=text_before, theme=theme)
                attempts += 1
            used_words.add(replacement)
            
            # Track the first noun as our "main character"
            if main_subject is None:
                main_subject = replacement
            else:
                # 20% chance to bring back the main subject for continuity
                # This can reintroduce the main subject for story continuity
                if random.random() < 0.2:
                    replacement = main_subject

        elif ph == "VERB":
            # Step 1: Figure out what tense we need
            tense, progressive = detect_verb_context(text_before)
            
            # Step 2: Pick base verb
            base = choose_ai_word_themed(VERBS, context, pos_type='VERB', 
                                        text_before=text_before, theme=theme)
            
            # Step 3: Avoid reusing
            attempts = 0
            while base in used_words and attempts < 15:
                base = choose_ai_word_themed(VERBS, context, pos_type='VERB', 
                                            text_before=text_before, theme=theme)
                attempts += 1
            used_words.add(base)
            
            # Step 4: Conjugate to proper form
            replacement = conjugate_verb(base, tense, progressive)

        elif ph == "ADJECTIVE":
            replacement = choose_ai_word_themed(ADJECTIVES, context, pos_type='ADJECTIVE', 
                                               text_before=text_before, theme=theme)
            
            # Avoid reusing
            attempts = 0
            while replacement in used_words and attempts < 15:
                replacement = choose_ai_word_themed(ADJECTIVES, context, pos_type='ADJECTIVE', 
                                                   text_before=text_before, theme=theme)
                attempts += 1
            used_words.add(replacement)

        else:
            # Unknown placeholder? Use blank
            replacement = "____"

        # Add to context for future semantic scoring
        context.append(replacement)
        
        # Replace in template (only first occurrence to handle duplicates correctly)
        filled = filled.replace(f"[{ph}]", replacement, 1)

    return filled

# ========================
# ENHANCED THEME MODE
# ========================
def choose_theme_mode():
    """
    Give the user three options:
    1. pick a specific theme  
    2. choose a random theme  
    3. use no theme

    Returns the selected theme name or None.
    """
    print("\n=== THEME SELECTION ===")
    print("1. Use a specific theme")
    print("2. Random theme (surprise me!)")
    print("3. No theme (use general words)")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            # Let user manually pick
            print("\nAvailable themes:")
            themes = list(THEMED_WORDS.keys())
            for i, theme in enumerate(themes, 1):
                print(f"{i}. {theme.capitalize()}")
            
            while True:
                try:
                    theme_choice = input(f"\nSelect theme (1-{len(themes)}): ").strip()
                    idx = int(theme_choice) - 1
                    if 0 <= idx < len(themes):
                        selected = themes[idx]
                        print(f"\n✨ Theme selected: {selected.upper()} ✨\n")
                        return selected
                    else:
                        print(f"Please enter a number between 1 and {len(themes)}")
                except ValueError:
                    print("Please enter a valid number")
        
        elif choice == "2":
            # Random theme - let fate decide
            selected = random.choice(list(THEMED_WORDS.keys()))
            print(f"\n🎲 Random theme selected: {selected.upper()} 🎲\n")
            return selected
        
        elif choice == "3":
            # No theme - classic mode
            print("\n📝 Using general words (no theme)\n")
            return None
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# ========================
# TEMPLATE SELECTION
# ========================
def choose_template():
    """
    Let the user choose between a random built-in template
    or entering their own.

    Custom templates use [NOUN], [VERB], [ADJECTIVE] markers.
    """
    print("\nChoose an option:")
    print("1. Random template")
    print("2. Enter my own template")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        return random.choice(TEMPLATES)
    elif choice == "2":
        return input("\nType your custom template with brackets around the POS you want (e.g. [NOUN]):\n")
    else:
        print("Invalid choice. Using random template.")
        return random.choice(TEMPLATES)

# ========================
# TEMPLATE FILLING (NON-THEMED)
# ========================

def fill_template(template):
    """
    Fill a template without theme support.

    Replaces each placeholder using choose_ai_word(),
    tracks context, and avoids repeating words.
    """
    placeholders = re.findall(r"\[(.*?)\]", template)
    filled = template
    context = []
    used_words = set()

    for ph in placeholders:
        # Get text before this placeholder
        text_before = filled.split(f"[{ph}]")[0]
        
        if ph == "NOUN":
            replacement = choose_ai_word(NOUNS, context, pos_type='NOUN', text_before=text_before)
            
            # Avoid reusing
            attempts = 0
            while replacement in used_words and attempts < 40:
        
                replacement = choose_ai_word(NOUNS, context, pos_type='NOUN', text_before=text_before)
                attempts += 1
            
            used_words.add(replacement)

        elif ph == "VERB":
            # Detect tense
            tense, progressive = detect_verb_context(text_before)
            
            # Choose base verb
            base = choose_ai_word(VERBS, context, pos_type='VERB', text_before=text_before)
            
            # Avoid reusing
            attempts = 0
            while base in used_words and attempts < 15:
                base = choose_ai_word(VERBS, context, pos_type='VERB', text_before=text_before)
                attempts += 1
            
            used_words.add(base)
            
            # Conjugate
            replacement = conjugate_verb(base, tense, progressive)

        elif ph == "ADJECTIVE":
            replacement = choose_ai_word(ADJECTIVES, context, pos_type='ADJECTIVE', text_before=text_before)
            
            # Avoid reusing
            attempts = 0
            while replacement in used_words and attempts < 15:
                replacement = choose_ai_word(ADJECTIVES, context, pos_type='ADJECTIVE', text_before=text_before)
                attempts += 1
            
            used_words.add(replacement)

        else:
            replacement = "____"

        # Track for context
        context.append(replacement)
        
        # Replace in template
        filled = filled.replace(f"[{ph}]", replacement, 1)

    return filled

# ========================
# MAIN FUNCTION
# ========================
def main():
    """
    Entry point for the program.

    Handles:
    - theme selection
    - template selection
    - showing the template
    - generating and printing the final story
    """
    # Step 1: Theme selection
    theme = choose_theme_mode() 
    
    # Step 2: Template selection
    template = choose_template()
    
    # Step 3: Show original template
    print("\n--- TEMPLATE ---")
    print(template)
    
    # Step 4: Fill with AI-selected words
    result = fill_template_themed(template, theme)
    
    # Step 5: Show final story
    print("\n--- YOUR MAD LIBS STORY ---")
    print(result)

# ========================
# RUN THE PROGRAM
# ========================
if __name__ == "__main__":
    main()