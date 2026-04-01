# AI-MadLibs-Project

**Abstract – This project aims to build an AI-powered Mad Libs generator that fills in story blanks using natural language processing and word-embedding-based semantic analysis. By combining part-of-speech tagging, context vector similarity, and AI scoring mechanisms, our system will choose words that are both grammatically correct and contextually meaningful. The result is a theme-based, coherent story generator that uses AI techniques to understand and generate language.**

main.py includes the core code of this project while analysis.py includes the code used to develop an analysis of the efficiency of the code.

First, the code asks users to choose what theme they want the words to relate to. The five choices available are Spooky, Romantic, Scifi, Adventure, and Comedy. The user can also choose to make the program use a random theme, or make the program not use a theme at all.

The program then gives users two options:

(1) Use a random template

(2) Use their own template

Our program implements two natural language processing (NLP) libraries – NLTK and spaCy – to enhance grammatical accuracy. We also use semantic scoring and a word selection algorithm to output a grammatically correct and semantically relevant completed MadLibs.

## How to Run the Program

### Requirements
- Python 3.9 or higher

### Dependencies
Install the required libraries using pip:

```bash
pip install nltk spacy matplotlib
```

Download the necessary NLTK data by running Python and executing:

```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

Download the required spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

### Project Files
- `main.py` – Core Mad Libs generator logic
- `analysis.py` – Runs semantic analysis and visualizes results
- `templates.py` – Contains default story templates
- `themes.py` – Contains themes and associated vocabulary
- `README.md` – Project documentation

Ensure all files are located in the same directory before running the program.

---

### Running the Program

To run the Mad Libs generator:

```bash
python main.py
```

Follow the prompts to:
1. Select a theme (or choose random/no theme)
2. Choose a random template or provide your own

To run the semantic analysis and generate plots:

```bash
python analysis.py
```

---

### Output
- `main.py` outputs a completed, theme-based Mad Libs story
- `analysis.py` computes semantic similarity scores and displays a matplotlib graph
