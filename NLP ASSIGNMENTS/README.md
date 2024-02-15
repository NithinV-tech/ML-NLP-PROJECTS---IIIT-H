## TOKENIZER

The `Tokenizer` class in the provided code performs a comprehensive set of text preprocessing and tokenization tasks. Here's a breakdown of its functionalities:

### 1. Abbreviation and Contraction Handling
- Expands common contractions and abbreviations ( "can't" to "cannot",they'd to they would etc.).

### 2. Special Token Replacement
 (a) Sentence Tokenizer: Divides text into sentences.
 (b) Word Tokenizer: Splits sentences into individual words.
 (c) Numbers: Identifies numerical values.
 (d) Mail IDs: Recognizes email addresses.
 (e) Punctuation: Detects punctuation marks.
 (f) URLs: Identifies website links.
 (g) Hashtags (#omg): Recognizes social media hashtags.
 (h) Mentions (@john): Recognizes social media mentions.
 (i) Replaces punctuation marks with <PUNC>
 (j) detects date and time
 (k) detects phone number
 (l) detects number

### 3. Punctuation and Special Characters Handling
- **Sentence Endings:** Identifies sentence-ending punctuation (., !, ?) and marks the end of sentences with `<eos>`.


### 4. Whitespace Normalization
- Replaces new lines (`\n`), carriage returns (`\r`), and tabs (`\t`) with a single space to ensure consistent whitespace.

### 5. Tokenization Functions
- **tokenize_q1:** Tokenizes the text into sentences and further into words, handling the special cases and replacements mentioned above.
- **tokenize_ngram:** Similar to `tokenize_q1`, but also adds `<sos>` at the beginning and `<eos>` at the end of each sentence to denote sentence start and end, useful for n-gram models.

### How to run Tokenizer?
```bash
python tokenizer.py
```
your text: Is that what you mean? I am unsure.
tokenized text: [['Is', 'this', 'what', 'you', 'mean', '?'],
['I', 'am', 'unsure', '.']]

## LANGUAGE MODEL

# ASSUMPTIONS
 - the unseen probability of good turing smoothing is taken as 1/1000 for better estimation of unseen items

### How to Run?
```bash
python Smoothing.py <lm_type> <corpus_path>
```

- **LM type:** Can be g for Good-Turing Smoothing Model and i for Interpolation Model.
- On running the file, the expected output is a prompt, which asks for a sentence and provides the perplexity of that sentence using the given mechanism. For example:

```bash
python Smoothing.py  ./corpus.txt
input sentence: I am a woman.
score: 69.092021
```

## Implemented Models:

### On “Pride and Prejudice” corpus:
- **LM 1:** Tokenization + 3-gram LM + Good-Turing Smoothing
- **LM 2:** Tokenization + 3-gram LM + Linear Interpolation

### On “Ulysses” corpus:
- **LM 3:** Tokenization + 3-gram LM + Good-Turing Smoothing
- **LM 4:** Tokenization + 3-gram LM + Linear Interpolation

# GENERATION
## How to Run?

```bash
python3 generator.py <lm_type> <corpus_path> <k>
```
- LM type: Can be g for Good-Turing Smoothing Model and i for Interpolation Model.
- k: Denotes the number of candidates for the next word to be printed.
- On running the file, the expected output is a prompt, which asks for a sentence and it would again ask for no of words to predicted  just to reconfirm and provides k next words.

```bash

python3 generator.py i ./corpus.txt 3
input sentence: An apple a day keeps the doctor
output: enter no of words to be predicted
away 0.4
happy 0.2
fresh 0.1
```



