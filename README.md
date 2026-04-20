# 🧠 Natural Language Processing (NLP) — Complete Notes
### From Absolute Beginner → Job-Ready (ML / DL / Gen AI / Agentic AI)

> **How to use this guide:** Read sequentially for best understanding. Each section builds on the previous. Code examples are runnable in Google Colab or Jupyter Notebook.

---

## 📚 Table of Contents

| # | Topic | Level |
|---|-------|-------|
| 1 | [What is NLP?](#1-what-is-nlp) | 🟢 Beginner |
| 2 | [NLP Pipeline Overview](#2-nlp-pipeline-overview) | 🟢 Beginner |
| 3 | [Text Preprocessing](#3-text-preprocessing) | 🟢 Beginner |
| 4 | [Tokenization](#4-tokenization) | 🟢 Beginner |
| 5 | [Stemming](#5-stemming) | 🟢 Beginner |
| 6 | [Lemmatization](#6-lemmatization) | 🟢 Beginner |
| 7 | [Stopwords](#7-stopwords) | 🟢 Beginner |
| 8 | [Regular Expressions in NLP](#8-regular-expressions-in-nlp) | 🟡 Intermediate |
| 9 | [Parts of Speech (POS) Tagging](#9-parts-of-speech-pos-tagging) | 🟡 Intermediate |
| 10 | [Named Entity Recognition (NER)](#10-named-entity-recognition-ner) | 🟡 Intermediate |
| 11 | [Text Vectorization Overview](#11-text-vectorization-overview) | 🟡 Intermediate |
| 12 | [One Hot Encoding (OHE)](#12-one-hot-encoding-ohe) | 🟡 Intermediate |
| 13 | [Bag of Words (BOW)](#13-bag-of-words-bow) | 🟡 Intermediate |
| 14 | [N-Grams](#14-n-grams) | 🟡 Intermediate |
| 15 | [TF-IDF](#15-tf-idf) | 🟡 Intermediate |
| 16 | [Word Embeddings & Word2Vec](#16-word-embeddings--word2vec) | 🔴 Advanced |
| 17 | [CBOW (Continuous Bag of Words)](#17-cbow-continuous-bag-of-words) | 🔴 Advanced |
| 18 | [Skip-Gram](#18-skip-gram) | 🔴 Advanced |
| 19 | [Avg Word2Vec](#19-avg-word2vec) | 🔴 Advanced |
| 20 | [GloVe Embeddings](#20-glove-embeddings) | 🔴 Advanced |
| 21 | [FastText](#21-fasttext) | 🔴 Advanced |
| 22 | [Cosine Similarity](#22-cosine-similarity) | 🔴 Advanced |
| 23 | [Recurrent Neural Networks (RNN)](#23-recurrent-neural-networks-rnn) | 🔴 Advanced |
| 24 | [LSTM & GRU](#24-lstm--gru) | 🔴 Advanced |
| 25 | [Attention Mechanism](#25-attention-mechanism) | 🔴 Advanced |
| 26 | [Transformers Architecture](#26-transformers-architecture) | 🔴 Advanced |
| 27 | [BERT](#27-bert) | 🔴 Advanced |
| 28 | [GPT Family](#28-gpt-family) | 🔴 Advanced |
| 29 | [Hugging Face & Pipelines](#29-hugging-face--pipelines) | 🔴 Advanced |
| 30 | [Gen AI & LLMs](#30-gen-ai--llms) | 🚀 Expert |
| 31 | [RAG (Retrieval Augmented Generation)](#31-rag-retrieval-augmented-generation) | 🚀 Expert |
| 32 | [Agentic AI & LangChain](#32-agentic-ai--langchain) | 🚀 Expert |
| 33 | [NLP Projects](#33-nlp-projects) | 🚀 Expert |
| 34 | [Interview Questions](#34-interview-questions) | 📋 All Levels |

---

## 1. What is NLP?

### Simple Definition
NLP (Natural Language Processing) is a field of AI that helps computers **understand, interpret, and generate human language** — just like how you understand this sentence.

### Real-World Examples
- 🔍 Google Search understands your query
- 📧 Gmail's spam filter
- 🤖 ChatGPT answers your questions
- 🎤 Siri/Alexa understand speech
- 🌐 Google Translate
- 💬 WhatsApp's autocorrect

### Why is NLP Hard?
```
"I saw the man with the telescope"
→ Did I use the telescope to see him?
→ Or did he have a telescope?
```
Human language is **ambiguous, contextual, and constantly evolving**.

### NLP Tasks (What can NLP do?)

| Task | Example |
|------|---------|
| **Sentiment Analysis** | "This movie is amazing" → Positive |
| **Text Classification** | Email → Spam / Not Spam |
| **Named Entity Recognition** | "Elon Musk founded Tesla" → Person, Organization |
| **Machine Translation** | English → Hindi |
| **Text Summarization** | Long article → Short summary |
| **Question Answering** | "Who is the CEO of Apple?" → Tim Cook |
| **Text Generation** | ChatGPT writes an essay |
| **Speech Recognition** | Audio → Text |

---

## 2. NLP Pipeline Overview

### Simple Definition
An NLP pipeline is the **step-by-step process** of transforming raw text into something a machine learning model can understand and learn from.

```
RAW TEXT
   ↓
[Step 1] Text Preprocessing (Clean the text)
   ↓
[Step 2] Feature Extraction (Convert text → numbers)
   ↓
[Step 3] ML / DL Model (Train & Predict)
   ↓
OUTPUT (Prediction / Generation)
```

### The Full Pipeline (From Your Notes)

```
DATASET
  ↓
Text Preprocessing - 1        Text Preprocessing - 2
  ① Tokenization               ① Stemming
  ② Lowercase the words        ② Lemmatization
  ③ Regular Expression         ③ Stopwords
  ↓
Text → Vectors (Feature Extraction)
  ① One Hot Encoding
  ② Bag of Words (BOW)
  ③ TF-IDF
  ④ Word2Vec
  ⑤ Avg Word2Vec
  ↓
ML Algorithms → OUTPUT
```

### Install Required Libraries
```python
# Install all required NLP libraries
pip install nltk spacy gensim scikit-learn numpy pandas

# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Download SpaCy model
# python -m spacy download en_core_web_sm
```

---

## 3. Text Preprocessing

### Simple Definition
Text preprocessing is **cleaning and preparing raw text** so that a machine can process it effectively. Just like how you wash vegetables before cooking.

### Why Preprocessing?
- Text from internet/users is messy: HTML tags, emojis, punctuation, uppercase/lowercase differences
- "Good", "GOOD", "good!" — machine treats these as 3 different words without preprocessing
- Reduces noise → better model accuracy

### Common Preprocessing Steps

```python
import re
import string

def preprocess_text(text):
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Step 4: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 5: Remove numbers (optional, depends on task)
    text = re.sub(r'\d+', '', text)
    
    # Step 6: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example
text = "Hello World! This is NLP 101. Visit https://example.com for more info."
print(preprocess_text(text))
# Output: "hello world this is nlp  visit  for more info"
```

### Complete Preprocessing Pipeline

```python
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

def full_preprocessing(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # 5. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

text = "The food was absolutely amazing! I loved every bite of the delicious pizza."
print(full_preprocessing(text))
# Output: "food absolutely amazing loved every bite delicious pizza"
```

---

## 4. Tokenization

### Simple Definition
Tokenization is the process of **splitting text into smaller units** called tokens. A token can be a word, sentence, or even a character.

```
"The food is good" → ["The", "food", "is", "good"]
```

### Types of Tokenization

| Type | Input | Output |
|------|-------|--------|
| **Word Tokenization** | "I love NLP" | ["I", "love", "NLP"] |
| **Sentence Tokenization** | "I love NLP. It is fun." | ["I love NLP.", "It is fun."] |
| **Character Tokenization** | "NLP" | ["N", "L", "P"] |
| **Subword Tokenization** | "unhappiness" | ["un", "happi", "ness"] |

### Code — Word Tokenization

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer

# Basic word tokenization
text = "Hello! My name is John. I love Natural Language Processing."

# 1. Word Tokenizer
word_tokens = word_tokenize(text)
print("Word Tokens:", word_tokens)
# ['Hello', '!', 'My', 'name', 'is', 'John', '.', 'I', 'love', 
#  'Natural', 'Language', 'Processing', '.']

# 2. Sentence Tokenizer
sent_tokens = sent_tokenize(text)
print("Sentence Tokens:", sent_tokens)
# ['Hello!', 'My name is John.', 'I love Natural Language Processing.']

# 3. Tweet Tokenizer (handles hashtags, emojis)
tweet = "I love #NLP! It's awesome 😊 @John"
tweet_tokenizer = TweetTokenizer()
print("Tweet Tokens:", tweet_tokenizer.tokenize(tweet))
# ['I', 'love', '#NLP', '!', "It's", 'awesome', '😊', '@John']
```

### Code — Different NLTK Tokenizers

```python
from nltk.tokenize import (
    word_tokenize,
    WordPunctTokenizer,
    RegexpTokenizer,
    MWETokenizer
)

text = "Don't hesitate to ask questions!"

# WordPunctTokenizer — splits on punctuation too
wp_tokenizer = WordPunctTokenizer()
print(wp_tokenizer.tokenize(text))
# ["Don", "'", "t", "hesitate", "to", "ask", "questions", "!"]

# RegexpTokenizer — custom pattern
regexp_tokenizer = RegexpTokenizer(r'\w+')  # only words, no punctuation
print(regexp_tokenizer.tokenize(text))
# ['Don', 't', 'hesitate', 'to', 'ask', 'questions']
```

### Code — SpaCy Tokenization (Industry Standard)

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

# Word tokens
for token in doc:
    print(f"Token: {token.text:15} | Lemma: {token.lemma_:15} | POS: {token.pos_}")
```

### Subword Tokenization (Used in BERT, GPT)

```python
# BPE (Byte Pair Encoding) — used in GPT
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "unhappiness is natural"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['un', 'happiness', 'Ġis', 'Ġnatural']

# WordPiece — used in BERT
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = bert_tokenizer.tokenize("unhappiness")
print(tokens)  # ['un', '##happiness']  (## = continuation of word)
```

> **Interview Tip:** BERT uses WordPiece, GPT uses BPE, T5 uses SentencePiece.

---

## 5. Stemming

### Simple Definition
Stemming is the process of **reducing a word to its base/root form** by chopping off the end of the word. It's a crude/fast approach.

```
"running" → "run"
"played"  → "play"
"studies" → "studi"  (may not be a real word!)
```

### Types of Stemmers

| Stemmer | Speed | Accuracy | Notes |
|---------|-------|----------|-------|
| **Porter Stemmer** | Fast | Medium | Most popular, aggressive |
| **Snowball Stemmer** | Fast | Better | Improved Porter |
| **Lancaster Stemmer** | Very Fast | Low | Most aggressive |
| **Regexp Stemmer** | Fast | Custom | You define the rules |

### Code — All Stemmer Types

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, RegexpStemmer

words = ["running", "played", "studies", "beautiful", "eating", 
         "happily", "goes", "going", "final", "finally"]

# 1. Porter Stemmer
porter = PorterStemmer()
print("=== Porter Stemmer ===")
for word in words:
    print(f"{word:15} → {porter.stem(word)}")

# 2. Snowball Stemmer (better than Porter)
snowball = SnowballStemmer("english")
print("\n=== Snowball Stemmer ===")
for word in words:
    print(f"{word:15} → {snowball.stem(word)}")

# 3. Lancaster Stemmer (most aggressive)
lancaster = LancasterStemmer()
print("\n=== Lancaster Stemmer ===")
for word in words:
    print(f"{word:15} → {lancaster.stem(word)}")

# 4. RegexpStemmer (custom rules)
regexp = RegexpStemmer('ing$|s$|e$|able$', min=4)
print("\n=== Regexp Stemmer ===")
for word in words:
    print(f"{word:15} → {regexp.stem(word)}")
```

### Output Comparison

```
Word            Porter    Snowball  Lancaster
running      →  run       run       run
played       →  play      play      play
studies      →  studi     studi     study
beautiful    →  beauti    beauti    beauty
happily      →  happili   happili   happy
```

### Disadvantages of Stemming
- ❌ May produce non-real words: "studies" → "studi"
- ❌ Same stem for different meanings: "better" → "bet" (wrong!)
- ❌ Over-stemming or under-stemming

> **When to use Stemming?** When speed matters more than accuracy, e.g., search engines, information retrieval.

---

## 6. Lemmatization

### Simple Definition
Lemmatization is **reducing a word to its base dictionary form** (called the lemma) using vocabulary and grammar rules. It always returns a real word.

```
"running" → "run"   (verb)
"better"  → "good"  (adjective)
"studies" → "study" (real word!)
"wolves"  → "wolf"
```

### Stemming vs Lemmatization

| | Stemming | Lemmatization |
|--|---------|---------------|
| **Speed** | Fast | Slower |
| **Output** | May not be real word | Always real word |
| **Uses Grammar?** | No | Yes (POS needed) |
| **Accuracy** | Lower | Higher |
| **Use case** | Search engines | Chatbots, NLP models |

### Code — NLTK Lemmatizer

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Lemmatize with POS tag for accuracy
words_pos = [
    ("running", "v"),   # v = verb
    ("better",  "a"),   # a = adjective
    ("studies", "v"),   # v = verb
    ("wolves",  "n"),   # n = noun
    ("ate",     "v"),
    ("gone",    "v"),
    ("happily", "r"),   # r = adverb
]

print(f"{'Word':15} {'POS':5} {'Lemma':15}")
print("-" * 35)
for word, pos in words_pos:
    lemma = lemmatizer.lemmatize(word, pos=pos)
    print(f"{word:15} {pos:5} {lemma:15}")
```

### Code — SpaCy Lemmatization (Industry Preferred)

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "The cats are running faster than the wolves who were eating"
doc = nlp(text)

for token in doc:
    print(f"Word: {token.text:15} | Lemma: {token.lemma_:15} | POS: {token.pos_}")
```

---

## 7. Stopwords

### Simple Definition
Stopwords are **common words that appear frequently but carry little meaning** — words like "is", "the", "a", "and". We remove them to focus on meaningful words.

```
"The food is very good" → after removing stopwords → "food good"
```

### Code — Stopwords with NLTK

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# See all English stopwords
stop_words = set(stopwords.words('english'))
print(f"Total stopwords: {len(stop_words)}")
print(sorted(stop_words)[:20])
# ['a', 'about', 'above', 'after', 'again', 'against', 'ain', ...]

# Remove stopwords from text
text = "The food at this restaurant is absolutely amazing and I love it"
tokens = word_tokenize(text.lower())
filtered = [word for word in tokens if word not in stop_words]

print("Original:", tokens)
print("Filtered:", filtered)
# Filtered: ['food', 'restaurant', 'absolutely', 'amazing', 'love']
```

### Code — Custom Stopwords

```python
from nltk.corpus import stopwords

# Add custom stopwords
custom_stop = set(stopwords.words('english'))
custom_stop.update(['also', 'would', 'could', 'said', 'get'])

# Remove custom stopwords
text = "I would also get some food if I could"
tokens = text.lower().split()
filtered = [w for w in tokens if w not in custom_stop]
print(filtered)  # ['food']

# Multiple languages
hindi_stopwords = stopwords.words('hindi')  # NLTK has many languages!
```

> ⚠️ **Note:** In sentiment analysis, don't remove "not", "never", "no" — they reverse meaning! "This is NOT good" → removing "not" → "This is good" (WRONG!)

---

## 8. Regular Expressions in NLP

### Simple Definition
Regular expressions (regex) are **patterns used to find, match, or replace specific text**. Very useful for cleaning text data.

### Cheat Sheet

| Pattern | Meaning | Example |
|---------|---------|---------|
| `\d` | Any digit | `\d+` matches "123" |
| `\w` | Word character | `\w+` matches "hello" |
| `\s` | Whitespace | `\s+` matches spaces |
| `^` | Start of string | `^Hello` |
| `$` | End of string | `world$` |
| `.*` | Any characters | |
| `[a-z]` | Character range | lowercase letters |
| `|` | OR operator | `cat|dog` |

### Code — Common NLP Regex Tasks

```python
import re

text = """Hello! Visit us at https://www.example.com or email us at
info@example.com. Call us: +1-800-555-0123. Today is 2024-01-15.
<b>Special offer!</b> Save 20% today!!! #NLP @nlpguru"""

# 1. Remove URLs
clean = re.sub(r'https?://\S+|www\.\S+', '', text)

# 2. Remove email addresses
clean = re.sub(r'\S+@\S+', '', clean)

# 3. Remove phone numbers
clean = re.sub(r'[\+\d][\d\-\(\)\s]{8,}', '', clean)

# 4. Remove HTML tags
clean = re.sub(r'<.*?>', '', clean)

# 5. Remove mentions and hashtags
clean = re.sub(r'@\w+|#\w+', '', clean)

# 6. Remove extra punctuation
clean = re.sub(r'[^\w\s]', '', clean)

# 7. Remove extra whitespace
clean = re.sub(r'\s+', ' ', clean).strip()

print(clean)
```

### Code — Extracting Information with Regex

```python
import re

text = "John's phone is 555-1234 and email is john@email.com, DOB: 1990-05-15"

# Extract phone numbers
phones = re.findall(r'\d{3}-\d{4}', text)
print("Phones:", phones)  # ['555-1234']

# Extract emails
emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
print("Emails:", emails)  # ['john@email.com']

# Extract dates
dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
print("Dates:", dates)  # ['1990-05-15']
```

---

## 9. Parts of Speech (POS) Tagging

### Simple Definition
POS tagging is **labeling each word in a sentence with its grammatical role** — noun, verb, adjective, etc.

```
"The quick brown fox jumps"
 DT   JJ    JJ    NN   VBZ
(Det)(Adj)(Adj)(Noun)(Verb)
```

### Common POS Tags

| Tag | Meaning | Example |
|-----|---------|---------|
| **NN** | Noun, singular | "dog", "car" |
| **NNS** | Noun, plural | "dogs", "cars" |
| **VB** | Verb, base | "run", "eat" |
| **VBG** | Verb, gerund | "running", "eating" |
| **JJ** | Adjective | "good", "fast" |
| **RB** | Adverb | "quickly", "very" |
| **DT** | Determiner | "the", "a" |
| **PRP** | Personal pronoun | "I", "he", "she" |
| **IN** | Preposition | "in", "on", "at" |

### Code — POS Tagging with NLTK

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

print(pos_tags)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'),
#  ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

# Filter only nouns
nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
print("Nouns:", nouns)  # ['fox', 'dog']

# Filter only verbs
verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
print("Verbs:", verbs)  # ['jumps']
```

### Code — POS Tagging with SpaCy (Better)

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying a U.K. startup for $1 billion"
doc = nlp(text)

for token in doc:
    print(f"{token.text:10} | {token.pos_:8} | {token.tag_:6} | {token.dep_}")

# Visualize (in Jupyter)
from spacy import displacy
displacy.render(doc, style="dep", jupyter=True)
```

### Real-World Use of POS Tagging
- Better lemmatization (need POS to know if "running" is verb or adjective)
- Feature extraction for ML models
- Named entity recognition
- Grammar checking

---

## 10. Named Entity Recognition (NER)

### Simple Definition
NER is the task of **identifying and classifying named entities** in text — like people, organizations, locations, dates, amounts, etc.

```
"Elon Musk founded Tesla in 2003 in California"
      PERSON      ORG          DATE    LOCATION
```

### Common Entity Types

| Label | Meaning | Example |
|-------|---------|---------|
| **PERSON** | People | "Barack Obama" |
| **ORG** | Organizations | "Google", "WHO" |
| **GPE** | Countries/Cities | "India", "New York" |
| **LOC** | Non-GPE locations | "Mount Everest" |
| **DATE** | Dates | "January 2024" |
| **MONEY** | Monetary values | "$1 billion" |
| **PRODUCT** | Products | "iPhone" |
| **EVENT** | Events | "World Cup" |

### Code — NER with NLTK

```python
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

text = "Mark Zuckerberg is the CEO of Meta, headquartered in Menlo Park, California."
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
ner_tree = ne_chunk(pos_tags)

# Extract named entities
for subtree in ner_tree:
    if isinstance(subtree, Tree):
        entity_name = ' '.join([token for token, pos in subtree.leaves()])
        entity_type = subtree.label()
        print(f"Entity: {entity_name:25} | Type: {entity_type}")
```

### Code — NER with SpaCy (Industry Standard)

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = """Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976. 
The company's revenue reached $394 billion in 2022."""

doc = nlp(text)

# Print named entities
for ent in doc.ents:
    print(f"Entity: {ent.text:20} | Label: {ent.label_:10} | Desc: {spacy.explain(ent.label_)}")

# Visualize (in Jupyter)
from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)
```

### Code — Custom NER Training with SpaCy

```python
# Training data format for custom NER
TRAIN_DATA = [
    ("iPhone 15 Pro was released by Apple", 
     {"entities": [(0, 13, "PRODUCT"), (30, 35, "ORG")]}),
    ("Samsung Galaxy S24 is available in India",
     {"entities": [(0, 18, "PRODUCT"), (35, 40, "GPE")]}),
]

# Training loop (simplified)
import spacy
from spacy.training import Example

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

# ... (full training requires more setup)
```

---

## 11. Text Vectorization Overview

### Simple Definition
Machines can't understand words directly — they only understand numbers. Text vectorization is the process of **converting text into numerical vectors** so ML algorithms can process it.

### Evolution of Text Vectorization

```
Generation 1 (Count-based)        Generation 2 (Neural)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━
① One Hot Encoding                  ④ Word2Vec (2013)
② Bag of Words (BOW)                ⑤ GloVe (2014)
③ TF-IDF                            ⑥ FastText (2016)
                                     ⑦ BERT Embeddings (2018)
```

### Comparison Table

| Method | Dense/Sparse | Captures Meaning? | OOV Problem? | Size |
|--------|-------------|------------------|--------------|------|
| OHE | Sparse | ❌ No | ✅ Yes | Vocab size |
| BOW | Sparse | ❌ No | ✅ Yes | Vocab size |
| TF-IDF | Sparse | Partial | ✅ Yes | Vocab size |
| Word2Vec | **Dense** | ✅ Yes | ❌ No | Fixed (300) |
| BERT | **Dense** | ✅✅ Yes | ❌ No | 768/1024 |

---

## 12. One Hot Encoding (OHE)

### Simple Definition
One Hot Encoding represents each word as a **vector of 0s with exactly one 1** at the position corresponding to that word in the vocabulary.

```
Vocabulary: ["The", "food", "is", "good", "bad", "Pizza", "Amazing"]
                0       1     2     3      4      5        6

"good" → [0, 0, 0, 1, 0, 0, 0]
"food" → [0, 1, 0, 0, 0, 0, 0]
```

### From Your Notes (Sentiment Analysis Example)

```
Dataset:
D1: "The food is good" → 1
D2: "The food is bad"  → 0
D3: "Pizza is Amazing" → 1
D4: "Burger is bad"    → 0

Vocabulary size: 7
["The", "food", "is", "good", "bad", "Pizza", "Amazing"]

D1 → [[1,0,0,0,0,0,0],   ← "The"
       [0,1,0,0,0,0,0],   ← "food"
       [0,0,1,0,0,0,0],   ← "is"
       [0,0,0,1,0,0,0]]   ← "good"
Shape: 4×7
```

### Code — OHE from Scratch

```python
import numpy as np

corpus = [
    "The food is good",
    "The food is bad",
    "Pizza is Amazing",
    "Burger is bad"
]

# Step 1: Build vocabulary
vocab = set()
for sentence in corpus:
    for word in sentence.lower().split():
        vocab.add(word)
vocab = sorted(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}

print("Vocabulary:", vocab)
print("Word to Index:", word2idx)

# Step 2: Create OHE vectors
def one_hot_encode(sentence, vocab, word2idx):
    tokens = sentence.lower().split()
    ohe_matrix = np.zeros((len(tokens), len(vocab)), dtype=int)
    for i, token in enumerate(tokens):
        if token in word2idx:
            ohe_matrix[i][word2idx[token]] = 1
    return ohe_matrix

# OHE for D1
d1_ohe = one_hot_encode(corpus[0], vocab, word2idx)
print(f"\nD1 OHE (shape {d1_ohe.shape}):")
print(d1_ohe)
```

### Code — Using Scikit-learn

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Word-level OHE
words = np.array([["the"], ["food"], ["is"], ["good"], ["bad"]])
encoder = OneHotEncoder(sparse_output=False)
ohe = encoder.fit_transform(words)
print(ohe)

# Using pandas get_dummies (easier)
import pandas as pd
words_df = pd.DataFrame({"word": ["the", "food", "is", "good", "bad"]})
ohe_df = pd.get_dummies(words_df["word"])
print(ohe_df)
```

### Advantages and Disadvantages

✅ **Advantages:**
- Simple to implement
- Can use sklearn's `OneHotEncoder` or `pd.get_dummies()`

❌ **Disadvantages:**
- **Sparse matrix** → huge memory → overfitting
- **Fixed size input required** for ML algorithms
- **No semantic meaning** captured ("good" and "great" are equally distant)
- **OOV (Out of Vocabulary)** problem: new words can't be encoded
- Size grows with vocabulary (50K words → 50K dimensions!)

---

## 13. Bag of Words (BOW)

### Simple Definition
BOW represents a document by **counting how many times each vocabulary word appears** in it — like putting words in a bag and counting each type.

```
S1: "He is a good boy"      → {good:1, boy:1, ...}
S2: "She is a good girl"    → {good:1, girl:1, ...}
S3: "Boy and girl are good" → {boy:1, girl:1, good:1, ...}

Vocabulary (after removing stopwords): [good, boy, girl]

       good  boy  girl
S1  →  [1,    1,   0]
S2  →  [1,    0,   1]
S3  →  [1,    1,   1]
```

### Binary BOW vs Regular BOW

```
Text: "I love love love NLP"

Binary BOW: {I:1, love:1, NLP:1}     ← just presence (0 or 1)
Regular BOW: {I:1, love:3, NLP:1}    ← actual count
```

### Code — BOW from Scratch

```python
from collections import Counter

corpus = [
    "He is a good boy",
    "She is a good girl",
    "Boy and girl are good"
]

# Step 1: Remove stopwords and build vocabulary
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = text.lower().split()
    return [t for t in tokens if t not in stop_words]

processed = [preprocess(s) for s in corpus]
print("Processed:", processed)

# Step 2: Build vocabulary (sorted by frequency)
all_words = [word for sent in processed for word in sent]
vocab = sorted(set(all_words))
word2idx = {word: i for i, word in enumerate(vocab)}
print("Vocabulary:", vocab)

# Step 3: Create BOW vectors
def bow_vector(tokens, word2idx):
    vec = [0] * len(word2idx)
    for token in tokens:
        if token in word2idx:
            vec[word2idx[token]] += 1
    return vec

for i, sent in enumerate(processed):
    vec = bow_vector(sent, word2idx)
    print(f"S{i+1}: {vec}")
```

### Code — BOW with Scikit-learn (Industry Way)

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "He is a good boy",
    "She is a good girl",
    "Boy and girl are good"
]

# CountVectorizer does: tokenization + lowercasing + stop words (optional)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BOW Matrix:\n", X.toarray())

# With stopwords removal
vectorizer_sw = CountVectorizer(stop_words='english')
X_sw = vectorizer_sw.fit_transform(corpus)
print("\nWith Stopwords Removed:")
print("Vocabulary:", vectorizer_sw.get_feature_names_out())
print("BOW Matrix:\n", X_sw.toarray())

# Binary BOW
binary_vec = CountVectorizer(binary=True)
X_binary = binary_vec.fit_transform(corpus)
print("\nBinary BOW:\n", X_binary.toarray())
```

### Advantages and Disadvantages

✅ **Advantages:**
- Simple and Intuitive
- Fixed Sized Input → ML Algorithms

❌ **Disadvantages:**
- Sparse matrix → Overfitting
- **Ordering of words is lost**: "good food" = "food good" (same BOW)
- OOV problem
- **Semantic meaning not captured**: "good" and "great" are different words

---

## 14. N-Grams

### Simple Definition
N-grams are **sequences of N consecutive words** from a text. They help capture word order and context — something BOW misses.

```
Text: "The food is good"
Unigrams (1-gram): ["The", "food", "is", "good"]
Bigrams (2-gram):  ["The food", "food is", "is good"]
Trigrams (3-gram): ["The food is", "food is good"]
```

### Why N-Grams?
BOW can't differentiate between "food is good" and "food is not good" because it ignores word order. N-grams capture context!

### From Your Notes

```
S1 → "The food is good"     (The, food circled as stopwords)
S2 → "The food is not good"

Bigrams from S1 & S2:
       food  not  good  food_good  food_not  not_good
S1  →   1     0    1      1           0          0
S2  →   1     1    1      0           1          1
```

### Code — N-Grams with NLTK

```python
from nltk import ngrams
from nltk.tokenize import word_tokenize

text = "The food is good at this restaurant"
tokens = word_tokenize(text.lower())

# Unigrams
unigrams = list(ngrams(tokens, 1))
print("Unigrams:", unigrams)

# Bigrams
bigrams = list(ngrams(tokens, 2))
print("Bigrams:", bigrams)

# Trigrams
trigrams = list(ngrams(tokens, 3))
print("Trigrams:", trigrams)
```

### Code — N-Gram BOW with CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "The food is good",
    "The food is not good",
    "Pizza is amazing"
]

# Unigram BOW
uni = CountVectorizer(ngram_range=(1,1))
print("Unigram features:", uni.fit(corpus).get_feature_names_out())

# Bigram BOW
bi = CountVectorizer(ngram_range=(2,2))
print("Bigram features:", bi.fit(corpus).get_feature_names_out())

# Unigram + Bigram (combined)
uni_bi = CountVectorizer(ngram_range=(1,2))
print("Uni+Bigram features:", uni_bi.fit(corpus).get_feature_names_out())

# Sklearn ngram_range=(1,3) means: unigram, bigram, trigram all together
all_grams = CountVectorizer(ngram_range=(1,3))
X = all_grams.fit_transform(corpus)
print("Shape:", X.shape)
```

### Scikit-learn N-gram Shorthand (from your notes)
```python
# Sklearn → n_grams = (1,1) → unigrams
#          = (1,2) → unigram, bigram
#          = (1,3) → unigram, bigram, trigram
#          = (2,3) → bigram, trigram
```

---

## 15. TF-IDF

### Simple Definition
TF-IDF (Term Frequency – Inverse Document Frequency) assigns **importance score to each word** based on:
- How often it appears in a document (TF)
- How rare it is across all documents (IDF)

Common words like "the", "is" get low scores. Rare, important words get high scores.

### The Math

```
TF(word, sentence) = Number of times word appears in sentence
                     ─────────────────────────────────────────
                     Total number of words in sentence

IDF(word) = log_e( Total number of sentences / Number of sentences containing the word )

TF-IDF = TF × IDF
```

### From Your Notes — Worked Example

```
S1: good boy        S2: good girl        S3: boy girl good
Vocabulary: {good, boy, girl}

Term Frequency:
           S1    S2    S3
good  →   1/2   1/2   1/3
boy   →   1/2    0    1/3
girl  →    0    1/2   1/3

IDF:
good → log(3/3) = log(1) = 0      ← appears in ALL docs, so 0 importance!
boy  → log(3/2) = 0.405
girl → log(3/2) = 0.405

Final TF-IDF:
           good         boy              girl
S1  →       0      1/2 × log(3/2)         0
S2  →       0           0           1/2 × log(3/2)
S3  →       0      1/3 × log(3/2)   1/3 × log(3/2)
```

### Code — TF-IDF from Scratch

```python
import numpy as np
import math

corpus = [
    "good boy",
    "good girl",
    "boy girl good"
]

# Tokenize
tokenized = [s.split() for s in corpus]

# Build vocabulary
vocab = sorted(set(w for s in tokenized for w in s))
word2idx = {w: i for i, w in enumerate(vocab)}

# Calculate TF
def compute_tf(tokens, vocab):
    tf = {}
    total = len(tokens)
    for word in vocab:
        tf[word] = tokens.count(word) / total
    return tf

# Calculate IDF
def compute_idf(corpus_tokens, vocab):
    N = len(corpus_tokens)
    idf = {}
    for word in vocab:
        docs_with_word = sum(1 for doc in corpus_tokens if word in doc)
        idf[word] = math.log(N / docs_with_word)
    return idf

# Calculate TF-IDF
idf = compute_idf(tokenized, vocab)

print(f"{'Word':8} {'IDF':8}")
for word, val in idf.items():
    print(f"{word:8} {val:.4f}")

tfidf_matrix = []
for tokens in tokenized:
    tf = compute_tf(tokens, vocab)
    row = [tf[word] * idf[word] for word in vocab]
    tfidf_matrix.append(row)

import pandas as pd
df = pd.DataFrame(tfidf_matrix, columns=vocab, index=['S1','S2','S3'])
print("\nTF-IDF Matrix:")
print(df.round(4))
```

### Code — TF-IDF with Scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "He is a good boy",
    "She is a good girl",
    "Boy and girl are good"
]

# Basic TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)

import pandas as pd
df = pd.DataFrame(X.toarray(), 
                  columns=tfidf.get_feature_names_out(),
                  index=['S1','S2','S3'])
print(df.round(3))

# With N-grams
tfidf_ngram = TfidfVectorizer(ngram_range=(1,2), 
                               stop_words='english',
                               max_features=1000)
X_ngram = tfidf_ngram.fit_transform(corpus)
print("Shape with N-grams:", X_ngram.shape)
```

### Advantages and Disadvantages

✅ **Advantages:**
1. Intuitive
2. Fixed Size → Vocab size
3. **Word Importance is captured** (rare words get higher score)

❌ **Disadvantages:**
1. Sparsity still exists
2. OOV problem

---

## 16. Word Embeddings & Word2Vec

### Simple Definition
Word embeddings represent words as **dense, low-dimensional vectors** where words with similar meanings are **close to each other** in vector space.

```
king  → [0.95, 0.96, ...]   (300 dimensions)
queen → [-0.96, 0.95, ...]
man   → [0.95, 0.98, ...]
woman → [-0.94, -0.96, ...]

KING - MAN + WOMAN ≈ QUEEN  ← Amazing analogy!
KING - BOY + QUEEN = GIRL   ← From your notes!
```

### OHE vs Word Embeddings

```
OHE (Sparse, 50K dims):
"king"  → [0,0,0,...,1,...,0,0]  ← 50,000 zeros with one 1

Word2Vec (Dense, 300 dims):
"king"  → [0.95, 0.96, -0.3, ...]  ← 300 meaningful numbers
```

### Word2Vec Definition
Word2Vec is a **neural network-based technique** (published by Google in 2013) that learns word representations from a large text corpus.

### Two Architectures

```
Word2Vec
    ├── CBOW (Continuous Bag of Words)  ← Context → Target word
    │   Use for: Small datasets
    └── Skip-gram                       ← Target word → Context
        Use for: Large datasets
```

### Feature Representation Example (From Your Notes)

```
           Boy    Girl   King   Queen  Apple  Mango
Gender    [-1      1    -0.92  +0.93   0.01   0.05]
Royal     [0.01   0.02   0.95   0.96  -0.02   0.02]
Age       [0.03   0.02   0.75   0.68   0.95   0.96]
Food      [ -      -      -      -     0.91   0.92]
```

### Code — Word2Vec with Gensim

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Sample corpus (use larger corpus in practice)
corpus = """
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.
It involves the interaction between computers and human language.
The goal is to program computers to process and analyze large amounts of natural language data.
Machine learning algorithms are used in NLP for tasks like sentiment analysis and translation.
Deep learning has revolutionized natural language processing in recent years.
"""

# Tokenize sentences and words
sentences = sent_tokenize(corpus)
tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,    # dimension of word vectors
    window=5,           # context window size
    min_count=1,        # ignore words with frequency < min_count
    workers=4,          # number of CPU cores
    sg=0,               # 0 = CBOW, 1 = Skip-gram
    epochs=100
)

# Save model
model.save("word2vec_model.bin")

# Load model
# model = Word2Vec.load("word2vec_model.bin")

print("Vocabulary size:", len(model.wv))
print("Vector for 'language':", model.wv['language'][:5], "...")
```

### Code — Using Google's Pretrained Word2Vec

```python
import gensim.downloader as api

# Download pretrained model (3 billion words, 300 dimensions)
# Warning: Large file ~1.6GB
model = api.load("word2vec-google-news-300")

# Find similar words
similar = model.most_similar("king", topn=5)
print("Similar to 'king':", similar)

# Word analogies: king - man + woman = ?
result = model.most_similar(positive=['king', 'woman'], 
                             negative=['man'], topn=3)
print("king - man + woman =", result)

# Cosine similarity
sim = model.similarity("good", "great")
print(f"Similarity between 'good' and 'great': {sim:.3f}")

sim2 = model.similarity("good", "bad")
print(f"Similarity between 'good' and 'bad': {sim2:.3f}")
```

### Advantages of Word2Vec

✅ **Advantages (from your notes):**
1. **Sparse → Dense Matrix** (no more sparsity!)
2. **Semantic Info is captured** (king - man + woman = queen)
3. **Fixed dimensions** (Google Word2Vec: 300 dimensions regardless of vocab size)
4. **OOV is also solved** (via subword models like FastText)

---

## 17. CBOW (Continuous Bag of Words)

### Simple Definition
CBOW predicts a **target word from its surrounding context words**. Given the context, predict the center word.

```
Window size = 5 (2 words on each side)

Corpus: "iNeuron Company Is Related To DATA SCIENCE"

Input (context):     iNeuron, Company, Related, To
Output (target):     IS  ← predict this!
```

### From Your Notes — Architecture

```
OHE vectors of context words (each 7×1)
    ↓
Input Layer (4 context word vectors: 7×1 each)
    ↓ (weights: 7×5)
Hidden Layer 1 (size = 5, window size)
    ↓ (weights: 5×7)
Output Layer (7×1 = vocab size)
    ↓
Softmax → Probability distribution
    ↓
Compare with actual OHE of target word
    ↓
Calculate Loss → Backpropagation → Update Weights
```

### Code — CBOW Data Preparation

```python
import numpy as np
from collections import Counter

def create_cbow_training_data(corpus, window_size=2):
    """Create (context, target) pairs for CBOW training"""
    tokens = corpus.split()
    vocab = sorted(set(tokens))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    
    training_data = []
    for i in range(window_size, len(tokens) - window_size):
        # Target word (center)
        target = tokens[i]
        
        # Context words (surrounding)
        context = (tokens[i-window_size:i] + 
                   tokens[i+1:i+window_size+1])
        
        training_data.append((context, target))
    
    return training_data, vocab, word2idx, idx2word

corpus = "iNeuron Company Is Related To DATA SCIENCE"
data, vocab, word2idx, idx2word = create_cbow_training_data(corpus, window_size=2)

print("Training pairs:")
for context, target in data[:3]:
    print(f"Context: {context} → Target: {target}")
```

### Code — CBOW with PyTorch (Neural Network)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context_ids):
        # Get embeddings for all context words
        embeds = self.embeddings(context_ids)  # (batch, context_size, embed_dim)
        
        # Average context embeddings
        avg_embed = embeds.mean(dim=1)  # (batch, embed_dim)
        
        # Project to vocab size
        out = self.linear(avg_embed)    # (batch, vocab_size)
        return out

# Using Gensim (practical approach)
from gensim.models import Word2Vec

sentences = [
    ["i", "love", "natural", "language", "processing"],
    ["nlp", "is", "a", "subfield", "of", "ai"],
    ["deep", "learning", "has", "improved", "nlp"],
]

# CBOW = sg=0
cbow_model = Word2Vec(sentences=sentences, vector_size=50, 
                       window=2, min_count=1, sg=0, epochs=200)
print("CBOW vector for 'nlp':", cbow_model.wv['nlp'][:5])
```

---

## 18. Skip-Gram

### Simple Definition
Skip-gram is the **opposite of CBOW**: given a target word, predict the surrounding context words.

```
Input: "Is"  (target)
Output: "Company", "Related", "To", "DATA"  (context)

vs CBOW:
Input: "Company", "Related", "To", "DATA"  (context)
Output: "Is"  (target)
```

### CBOW vs Skip-gram

| | CBOW | Skip-gram |
|--|------|-----------|
| **Direction** | Context → Target | Target → Context |
| **Dataset** | Small datasets | Large datasets |
| **Speed** | Faster | Slower |
| **Rare words** | Struggles | Better at rare words |
| **Use when** | Smaller corpus | Huge corpus (billions) |

### From Your Notes
```
Small Dataset  → CBOW
Huge Dataset   → Skip-gram

Google Word2Vec:
- 3 billion words → Google News
- 300 dimension vectors
- Cricket → [---. ......--.] (300 numbers)
```

### Code — Skip-gram with Gensim

```python
from gensim.models import Word2Vec

# Skip-gram = sg=1
skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=100,   # 300 in Google's model
    window=5,
    min_count=1,
    sg=1,              # 1 = Skip-gram, 0 = CBOW
    workers=4,
    epochs=100
)

# Compare CBOW vs Skip-gram similarities
words_to_test = [("good", "great"), ("good", "bad"), ("king", "queen")]
for w1, w2 in words_to_test:
    try:
        cbow_sim = cbow_model.wv.similarity(w1, w2)
        sg_sim = skipgram_model.wv.similarity(w1, w2)
        print(f"{w1}-{w2}: CBOW={cbow_sim:.3f}, Skip-gram={sg_sim:.3f}")
    except KeyError:
        pass
```

### How to Improve CBOW / Skip-gram (from your notes)
```
1. Increasing the Training Data
2. Increase the window size
   → vector dimension also increases
```

---

## 19. Avg Word2Vec

### Simple Definition
Since Word2Vec gives vectors for **individual words**, we need to represent an entire **sentence/document** as a single vector. Average Word2Vec does this by **averaging the word vectors** of all words in a sentence.

```
"The food is good"
→ Word2Vec("The") + Word2Vec("food") + Word2Vec("is") + Word2Vec("good")
  ────────────────────────────────────────────────────────────────────────
                                    4
= One 300-dimensional vector for the whole sentence
```

### From Your Notes
```
D1: "The food is good" → avg[The, food, is, good] → [-, -, -, -, -, -, -] (300 dims) → O/P: 1
D2: "The food is bad"  → avg[...] → [                                     ] → O/P: 0
D3: "Pizza is Amazing" → avg[...] → [                                     ] → O/P: 1

Google pretrained Word2Vec → Avg each word → 300-dim sentence vector
```

### Code — Avg Word2Vec for Sentiment Analysis

```python
import numpy as np
import gensim.downloader as api
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load pretrained model
print("Loading Google Word2Vec...")
w2v_model = api.load("word2vec-google-news-300")

def avg_word2vec(sentence, model, vector_size=300):
    """Get average word2vec for a sentence"""
    words = sentence.lower().split()
    valid_vectors = []
    
    for word in words:
        if word in model:
            valid_vectors.append(model[word])
    
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Sample dataset
texts = [
    "The food is absolutely amazing",
    "The food is terrible and bad",
    "Pizza is wonderful and delicious",
    "Burger is disgusting and horrible",
    "Service was excellent and friendly",
    "Very poor service and rude staff"
]
labels = [1, 0, 1, 0, 1, 0]

# Create feature vectors
X = np.array([avg_word2vec(text, w2v_model) for text in texts])
y = np.array(labels)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Code — Avg Word2Vec with Gensim (Practical)

```python
from gensim.models import Word2Vec
import numpy as np

# Train on your own corpus
sentences = [
    ["the", "food", "is", "good"],
    ["the", "food", "is", "bad"],
    ["pizza", "is", "amazing"],
    ["burger", "is", "terrible"]
]

model = Word2Vec(sentences=sentences, vector_size=100, 
                 window=5, min_count=1, epochs=200)

def get_avg_vector(sentence, model, vector_size=100):
    tokens = sentence.lower().split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vector_size)

# Test
for text in ["the food is good", "pizza is amazing"]:
    vec = get_avg_vector(text, model)
    print(f"'{text}' → vector shape: {vec.shape}, first 5: {vec[:5]}")
```

---

## 20. GloVe Embeddings

### Simple Definition
GloVe (Global Vectors for Word Representation) is **Stanford's word embedding method** (2014) that learns word vectors by analyzing **global word co-occurrence statistics** across the entire corpus.

```
Word2Vec: Learns from local context windows
GloVe:    Learns from global co-occurrence statistics (better!)
```

### Code — Using Pretrained GloVe

```python
import gensim.downloader as api

# Download GloVe (multiple sizes available)
# glove-twitter-25, glove-twitter-50, glove-twitter-100
# glove-wiki-gigaword-50, glove-wiki-gigaword-100, glove-wiki-gigaword-300

glove_model = api.load("glove-wiki-gigaword-100")

# Word similarity
print(glove_model.most_similar("python", topn=5))
print(glove_model.most_similar("king", topn=5))

# Analogy
print(glove_model.most_similar(positive=["king","woman"], negative=["man"]))
```

### Code — Load GloVe from File (Manual)

```python
import numpy as np

def load_glove_vectors(glove_file):
    """Load GloVe word vectors from text file"""
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Download from: https://nlp.stanford.edu/projects/glove/
# glove.6B.zip → glove.6B.100d.txt
# glove_vectors = load_glove_vectors('glove.6B.100d.txt')

def sentence_to_glove_avg(sentence, glove_vectors, dim=100):
    tokens = sentence.lower().split()
    vectors = [glove_vectors[w] for w in tokens if w in glove_vectors]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(dim)
```

---

## 21. FastText

### Simple Definition
FastText (by Facebook AI, 2016) extends Word2Vec by **treating each word as a bag of character n-grams**. This solves the OOV problem — it can create vectors even for words not seen during training!

```
Word: "eating"
Character n-grams (n=3): "<ea", "eat", "ati", "tin", "ing", "ng>"
FastText vector = sum of all character n-gram vectors
```

### Code — FastText with Gensim

```python
from gensim.models import FastText
import numpy as np

sentences = [
    ["i", "love", "natural", "language", "processing"],
    ["nlp", "is", "fascinating", "and", "useful"],
    ["machine", "learning", "drives", "modern", "nlp"],
]

# Train FastText
ft_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=3,
    min_count=1,
    sg=1,          # Skip-gram
    epochs=100
)

# FastText can handle OOV words!
print("Vector for 'nlp':", ft_model.wv['nlp'][:5])
print("Vector for 'nlpppp' (OOV):", ft_model.wv['nlpppp'][:5])  # Works!

# Compare with Word2Vec (Word2Vec would crash on OOV)
similar = ft_model.wv.most_similar("language", topn=3)
print("Similar to 'language':", similar)
```

---

## 22. Cosine Similarity

### Simple Definition
Cosine similarity measures **how similar two vectors are** by calculating the cosine of the angle between them. Value ranges from -1 to 1.

```
Cosine Similarity = 1   → Identical (0° angle)
Cosine Similarity = 0   → No relation (90° angle)
Cosine Similarity = -1  → Opposite (180° angle)

Distance = 1 - Cosine Similarity
```

### From Your Notes

```
If angle = 45°:
Cosine Similarity = cos(45°) = 1/√2 = 0.7071
Distance = 1 - 0.7071 = 0.29

If angle = 90° (IronMan vs Avengers on different axes):
Distance = 1 - cos(90°) = 1 - 0 = 1  (maximum distance)

If angle = 0° (same direction):
Distance = 1 - cos(0°) = 1 - 1 = 0   (no distance = identical)
```

### Code — Cosine Similarity

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(v1, v2):
    """Manual cosine similarity"""
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude == 0:
        return 0
    return dot_product / magnitude

# Example with word vectors
v1 = np.array([1, 2, 3, 4])
v2 = np.array([1, 2, 3, 4])  # identical
v3 = np.array([4, 3, 2, 1])  # different

print(f"v1 vs v2 (identical): {cosine_sim(v1, v2):.4f}")   # 1.0
print(f"v1 vs v3 (different): {cosine_sim(v1, v3):.4f}")   # ~0.8

# Using sklearn
print(cosine_similarity([v1], [v2]))  # [[1.0]]
print(cosine_similarity([v1], [v3]))  # [[0.8]]

# Document similarity with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "I love machine learning",
    "I love deep learning",
    "Python is a programming language"
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)
sims = cosine_similarity(X)

print("\nDocument Similarity Matrix:")
for i in range(len(docs)):
    for j in range(len(docs)):
        print(f"  Doc{i+1} vs Doc{j+1}: {sims[i][j]:.3f}")
```

---

## 23. Recurrent Neural Networks (RNN)

### Simple Definition
RNN is a type of neural network designed for **sequential data** (text, time series). Unlike regular NNs, RNNs have **memory** — the output at each step depends on previous steps.

```
Regular NN:   Input → Hidden → Output  (no memory)

RNN:          Input₁ → [Hidden₁] → Output₁
                              ↓
              Input₂ → [Hidden₂] → Output₂
                              ↓
              Input₃ → [Hidden₃] → Output₃
```

### Code — Simple RNN with Keras

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sentiment Analysis with RNN
texts = [
    "This movie is amazing",
    "Terrible film, waste of time",
    "Loved every moment of it",
    "Boring and predictable story"
]
labels = [1, 0, 1, 0]

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)
y = np.array(labels)

vocab_size = len(tokenizer.word_index) + 1

# Build RNN model
model = Sequential([
    Embedding(vocab_size, 32, input_length=10),
    SimpleRNN(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=50, verbose=0)

# Predict
test = tokenizer.texts_to_sequences(["This is wonderful"])
test_padded = pad_sequences(test, maxlen=10)
print("Prediction:", model.predict(test_padded))
```

### Problem with RNN: Vanishing Gradient
For long sequences, gradients become very small → RNN forgets long-range dependencies. **Solution: LSTM & GRU**

---

## 24. LSTM & GRU

### Simple Definition
**LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** are advanced versions of RNN that use **gates** to control what information to remember and forget. They solve the vanishing gradient problem.

### LSTM Gates
```
Forget Gate:  Decides what to FORGET from previous state
Input Gate:   Decides what NEW info to store
Output Gate:  Decides what to OUTPUT
```

### Code — LSTM for Text Classification

```python
from tensorflow.keras.layers import LSTM, GRU, Bidirectional

# LSTM Model
lstm_model = Sequential([
    Embedding(vocab_size, 128, input_length=100),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Bidirectional LSTM (reads text forward AND backward — better!)
bilstm_model = Sequential([
    Embedding(vocab_size, 128, input_length=100),
    Bidirectional(LSTM(64)),      # reads left-to-right AND right-to-left
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Stacked LSTM
stacked_lstm = Sequential([
    Embedding(vocab_size, 128, input_length=100),
    LSTM(128, return_sequences=True),   # return_sequences=True for stacking
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# GRU (simpler, faster than LSTM, similar performance)
gru_model = Sequential([
    Embedding(vocab_size, 128, input_length=100),
    GRU(128),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()
```

### When to use LSTM vs GRU?

| | LSTM | GRU |
|--|------|-----|
| **Complexity** | Higher | Lower |
| **Parameters** | More | Fewer |
| **Speed** | Slower | Faster |
| **Performance** | Slightly better on long sequences | Similar |
| **Use when** | Long sequences, complex tasks | Faster training needed |

---

## 25. Attention Mechanism

### Simple Definition
The Attention mechanism allows a model to **focus on relevant parts of the input** when generating each output word, instead of relying on just the last hidden state.

```
Translation: "The cat sat on the mat"
When generating "chat" (French for cat), 
the model ATTENDS to "cat" in the English input.
```

### Why Attention?
- Encoder-Decoder models compress entire input into one fixed vector → information bottleneck!
- With attention: each output step can look at ALL input steps and choose which to focus on

### Code — Self-Attention (Simplified)

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def scaled_dot_product_attention(Q, K, V, d_k):
    """
    Q = Query matrix
    K = Key matrix  
    V = Value matrix
    d_k = dimension of keys
    """
    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores)
    
    # Weighted sum of values
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

# Example
d_k = 4
Q = np.random.randn(3, d_k)  # 3 tokens, 4-dim queries
K = np.random.randn(3, d_k)  # 3 tokens, 4-dim keys
V = np.random.randn(3, d_k)  # 3 tokens, 4-dim values

output, weights = scaled_dot_product_attention(Q, K, V, d_k)
print("Attention Weights:\n", weights.round(3))
print("Output:\n", output.round(3))
```

---

## 26. Transformers Architecture

### Simple Definition
The Transformer is a neural network architecture (2017, "Attention is All You Need") that uses **only attention mechanisms** — no recurrence (no RNN/LSTM). It's the foundation of all modern LLMs (BERT, GPT, T5, etc.)

### Architecture Components

```
Transformer (Encoder-Decoder)
├── ENCODER (Input understanding)
│   ├── Input Embedding
│   ├── Positional Encoding  ← gives position info (no recurrence!)
│   └── N × Encoder Block
│       ├── Multi-Head Self-Attention
│       ├── Add & Norm
│       ├── Feed Forward Network
│       └── Add & Norm
│
└── DECODER (Output generation)
    ├── Output Embedding
    ├── Positional Encoding
    └── N × Decoder Block
        ├── Masked Multi-Head Self-Attention
        ├── Add & Norm
        ├── Cross-Attention (attends to encoder output)
        ├── Add & Norm
        ├── Feed Forward Network
        └── Add & Norm
```

### Code — Transformer with PyTorch

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Using PyTorch's built-in Transformer
model = nn.Transformer(
    d_model=512,          # model dimension
    nhead=8,              # number of attention heads
    num_encoder_layers=6, # number of encoder layers
    num_decoder_layers=6, # number of decoder layers
    dim_feedforward=2048, # feedforward network size
    dropout=0.1
)

print(f"Transformer parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 27. BERT

### Simple Definition
BERT (Bidirectional Encoder Representations from Transformers, by Google 2018) is a **pre-trained transformer model** that understands language context from **both left and right** simultaneously.

```
"I went to the bank to deposit money"  → bank = financial institution
"I sat on the river bank"               → bank = riverbank

BERT understands the difference because it reads BIDIRECTIONALLY!
```

### BERT vs GPT

| | BERT | GPT |
|--|------|-----|
| **Direction** | Bidirectional | Left-to-right only |
| **Architecture** | Encoder only | Decoder only |
| **Best for** | Understanding tasks | Generation tasks |
| **Tasks** | Classification, NER, QA | Text generation |
| **Pre-training** | MLM + NSP | Next token prediction |

### Code — BERT with Hugging Face

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

# ========================
# 1. Sentiment Analysis with BERT
# ========================
sentiment_pipeline = pipeline("sentiment-analysis", 
                               model="nlptown/bert-base-multilingual-uncased-sentiment")

texts = ["This movie was absolutely fantastic!", 
         "Terrible experience, never going back."]
for text in texts:
    result = sentiment_pipeline(text)
    print(f"Text: {text}")
    print(f"Result: {result}\n")

# ========================
# 2. Masked Language Modeling (BERT's pre-training task)
# ========================
unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("The [MASK] is the powerhouse of the cell.")
for r in result[:3]:
    print(f"Prediction: {r['token_str']:15} (score: {r['score']:.3f})")

# ========================
# 3. Feature Extraction with BERT
# ========================
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Natural language processing is amazing"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# CLS token = sentence representation
cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)
print(f"BERT sentence embedding shape: {cls_embedding.shape}")

# All token embeddings
token_embeddings = outputs.last_hidden_state  # shape: (1, seq_len, 768)
print(f"All token embeddings shape: {token_embeddings.shape}")
```

### Fine-tuning BERT for Classification

```python
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Prepare data
texts = ["Amazing product!", "Terrible quality!", "Good value", "Poor design"]
labels = [1, 0, 1, 0]

# Tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', 
                     truncation=True, max_length=128)

# Create dataset
dataset = Dataset.from_dict({'text': texts, 'label': labels})
tokenized_dataset = dataset.map(tokenize, batched=True)

# Fine-tune BERT
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```

---

## 28. GPT Family

### Simple Definition
GPT (Generative Pre-trained Transformer) is a **decoder-only transformer** trained to predict the next word. The GPT family (GPT-1 → GPT-4 → GPT-4o) from OpenAI has revolutionized text generation.

### GPT Evolution

```
GPT-1 (2018): 117M params | 1 book dataset
GPT-2 (2019): 1.5B params | 40GB text
GPT-3 (2020): 175B params | 570GB text → Few-shot learning
GPT-3.5:     + RLHF → ChatGPT (2022)
GPT-4:       Multimodal (text + images)
GPT-4o:      Real-time voice + vision
```

### Code — Using GPT-2 with Hugging Face

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Text generation with GPT-2
generator = pipeline("text-generation", model="gpt2")
result = generator(
    "Natural language processing is",
    max_length=50,
    num_return_sequences=2,
    temperature=0.7
)

for i, r in enumerate(result):
    print(f"Generated {i+1}: {r['generated_text']}\n")

# Using OpenAI API (GPT-4)
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an NLP expert."},
        {"role": "user", "content": "Explain what TF-IDF is in simple terms."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

---

## 29. Hugging Face & Pipelines

### Simple Definition
Hugging Face is the **GitHub of ML models** — a platform with 500,000+ pre-trained models for NLP, computer vision, audio, and more. The `transformers` library makes using these models easy.

### Quick Start with Pipelines

```python
from transformers import pipeline

# 1. Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
print(sentiment("I love Hugging Face!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 2. Text Generation
generator = pipeline("text-generation", model="gpt2")
print(generator("AI is transforming", max_length=50))

# 3. Named Entity Recognition
ner = pipeline("ner", aggregation_strategy="simple")
print(ner("Elon Musk founded Tesla in California"))

# 4. Question Answering
qa = pipeline("question-answering")
context = "Python was created by Guido van Rossum in 1991."
question = "Who created Python?"
print(qa(question=question, context=context))
# {'answer': 'Guido van Rossum', 'score': 0.99}

# 5. Text Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
long_text = """Natural language processing (NLP) is a subfield of linguistics, 
computer science, and artificial intelligence concerned with the interactions 
between computers and human language, in particular how to program computers 
to process and analyze large amounts of natural language data..."""
print(summarizer(long_text, max_length=50, min_length=20))

# 6. Translation
translator = pipeline("translation_en_to_fr")
print(translator("Hello, how are you today?"))

# 7. Zero-shot classification (no training needed!)
classifier = pipeline("zero-shot-classification")
text = "The stock market crashed today"
labels = ["finance", "sports", "politics", "technology"]
print(classifier(text, candidate_labels=labels))

# 8. Text to text generation
t5 = pipeline("text2text-generation", model="google/flan-t5-base")
print(t5("Translate English to French: Hello world"))
```

### Using Models Directly (AutoModel)

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Any model from Hugging Face Hub
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Inference
import torch

text = "This product is absolutely wonderful!"
inputs = tokenizer(text, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
prediction = torch.argmax(logits, dim=1)
print(f"Prediction: {model.config.id2label[prediction.item()]}")
```

---

## 30. Gen AI & LLMs

### Simple Definition
Generative AI (Gen AI) refers to AI systems that can **create new content** — text, images, code, audio. Large Language Models (LLMs) are the text-based Gen AI systems with billions of parameters trained on massive datasets.

### Key LLMs

| Model | Company | Parameters | Open Source? |
|-------|---------|-----------|--------------|
| GPT-4 | OpenAI | ~1.8T | ❌ |
| Claude 3.5 | Anthropic | Unknown | ❌ |
| Gemini Ultra | Google | Unknown | ❌ |
| Llama 3 70B | Meta | 70B | ✅ |
| Mistral 7B | Mistral AI | 7B | ✅ |
| Falcon 180B | TII | 180B | ✅ |

### Code — Using Ollama (Run LLMs Locally)

```python
# Install: https://ollama.ai
# ollama pull llama3

import ollama

response = ollama.chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': 'Explain word embeddings in simple terms'}
    ]
)
print(response['message']['content'])
```

### Prompt Engineering

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY")

# 1. Zero-shot
zero_shot = "Classify this review as positive or negative: 'The food was amazing!'"

# 2. Few-shot (give examples)
few_shot = """
Classify the review as positive or negative.

Review: "The pizza was amazing!" → Positive
Review: "Terrible service, never coming back" → Negative
Review: "Good value for money, decent food" → ?
"""

# 3. Chain-of-Thought (step-by-step reasoning)
cot = """
Review: "The food was okay but the service was terrible and the prices are too high."

Think step by step:
1. What is the sentiment about food? 
2. What is the sentiment about service?
3. What is the sentiment about price?
4. Overall sentiment?
"""

# 4. System Prompt
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert NLP researcher. Always be precise and technical."},
        {"role": "user", "content": "What's the difference between BERT and GPT?"}
    ]
)
```

---

## 31. RAG (Retrieval Augmented Generation)

### Simple Definition
RAG combines **information retrieval** with **text generation**. Instead of relying only on the LLM's training data, RAG fetches **relevant documents** from a knowledge base and uses them as context for the LLM.

```
Traditional LLM:
User Question → LLM → Answer (may hallucinate!)

RAG:
User Question → [Retrieve relevant docs from DB] → 
LLM (Question + Retrieved Docs) → Grounded Answer ✅
```

### RAG Pipeline

```
1. INDEXING (one-time setup):
   Documents → Chunk → Embed → Store in Vector DB

2. RETRIEVAL (at query time):
   User Query → Embed → Search Vector DB → Get Top-K chunks

3. GENERATION:
   Query + Retrieved chunks → LLM → Answer
```

### Code — Simple RAG with LangChain

```python
# pip install langchain langchain-openai faiss-cpu pypdf

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Step 1: Load documents
loader = TextLoader("my_document.txt")
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Step 4: Create RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Step 5: Query
result = qa_chain.invoke("What is the main topic of this document?")
print("Answer:", result["result"])
print("Sources:", result["source_documents"])
```

### RAG with Open Source (ChromaDB + Ollama)

```python
# pip install chromadb langchain-community

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Use local embeddings and LLM (no API key needed!)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = Ollama(model="llama3")

# Create vector store
vectorstore = Chroma.from_texts(
    texts=["Your documents here..."],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
print(qa.invoke("Your question here"))
```

---

## 32. Agentic AI & LangChain

### Simple Definition
Agentic AI refers to AI systems that can **plan, reason, use tools, and take actions autonomously** to accomplish complex goals — going beyond simple Q&A.

```
Traditional Chatbot:  User → LLM → Response
AI Agent:             User → LLM (thinks) → Uses Tool → LLM (thinks) → Answer
```

### LangChain — The Agent Framework

```python
# pip install langchain langchain-openai duckduckgo-search

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub

# Custom tool
@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate BMI given weight in kg and height in meters"""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5: category = "Underweight"
    elif bmi < 25: category = "Normal"
    elif bmi < 30: category = "Overweight"
    else: category = "Obese"
    return f"BMI: {bmi:.2f} ({category})"

# Define tools
search = DuckDuckGoSearchRun()
tools = [search, calculate_bmi]

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent
result = agent_executor.invoke({
    "input": "What is the latest news about AI? Also, calculate BMI for 70kg, 1.75m"
})
print(result["output"])
```

### Multi-Agent System with LangGraph

```python
# pip install langgraph

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]
    next_agent: str

def research_agent(state: AgentState):
    """Agent that researches topics"""
    # ... research logic
    return {"messages": state["messages"] + ["Research done"], "next_agent": "writer"}

def writer_agent(state: AgentState):
    """Agent that writes content"""
    # ... writing logic
    return {"messages": state["messages"] + ["Article written"], "next_agent": END}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_agent)
workflow.add_node("writer", writer_agent)
workflow.add_edge("researcher", "writer")
workflow.set_entry_point("researcher")

app = workflow.compile()
result = app.invoke({"messages": ["Write about NLP trends"], "next_agent": ""})
```

---

## 33. NLP Projects

### Project 1: Spam/Ham SMS Classifier

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# ── 1. Load Data ──
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print(df.shape)
print(df['label'].value_counts())

# ── 2. Preprocessing ──
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['processed'] = df['message'].apply(preprocess)

# ── 3. Feature Extraction ──
X = df['processed']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ── 4. Train Multiple Models ──
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear', probability=True)
}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))

# ── 5. Predict on new text ──
best_model = models['Naive Bayes']
new_sms = ["Congratulations! You won $1000! Click here to claim",
           "Hey, are we meeting for lunch tomorrow?"]
processed_new = [preprocess(s) for s in new_sms]
vec_new = tfidf.transform(processed_new)
predictions = best_model.predict(vec_new)
for sms, pred in zip(new_sms, predictions):
    print(f"'{sms}' → {'SPAM' if pred == 1 else 'HAM'}")
```

### Project 2: Kindle Review Sentiment Analysis (LSTM)

```python
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── 1. Load Data ──
df = pd.read_csv('kindle_reviews.csv')
# Assume columns: 'reviewText', 'overall' (1-5 stars)
df['sentiment'] = (df['overall'] >= 4).astype(int)  # 4-5 stars = positive

# ── 2. Preprocessing ──
import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['cleaned'] = df['reviewText'].apply(clean_text)

# ── 3. Tokenization & Padding ──
MAX_WORDS = 20000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned'])

X = tokenizer.texts_to_sequences(df['cleaned'])
X = pad_sequences(X, maxlen=MAX_LEN, padding='post', truncating='post')
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 4. Build BiLSTM Model ──
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ── 5. Train ──
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

# ── 6. Evaluate ──
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))
```

### Project 3: Text Summarization

```python
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

# Using pre-trained BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language, in particular how to program computers to process and analyze 
large amounts of natural language data. The goal is a computer capable of 
understanding the contents of documents, including the contextual nuances of 
the language within them. The technology can then accurately extract information 
and insights contained in the documents, as well as categorize and organize the 
documents themselves. Challenges in natural language processing frequently involve 
speech recognition, natural-language understanding, and natural-language generation.
"""

summary = summarizer(article, max_length=100, min_length=30, do_sample=False)
print("Summary:", summary[0]['summary_text'])
```

---

## 34. Interview Questions

### 🟢 Beginner Level

**Q: What is the difference between Stemming and Lemmatization?**
> Stemming chops off word endings using rules (fast but may produce non-words: "studies"→"studi"). Lemmatization uses vocabulary and grammar to return the actual base word (slower but accurate: "studies"→"study").

**Q: What is TF-IDF and why is it better than BOW?**
> TF-IDF assigns weights to words based on how important they are — common words (the, is) get low scores, rare important words get high scores. BOW just counts frequency, treating all words equally.

**Q: What is OOV problem and how to solve it?**
> OOV (Out of Vocabulary) = model encounters a word not in training vocabulary. Solutions: FastText (character n-grams), subword tokenization (BPE, WordPiece), or using pre-trained embeddings.

### 🟡 Intermediate Level

**Q: Explain CBOW vs Skip-gram.**
> CBOW predicts target word from context words (faster, better for frequent words, small datasets). Skip-gram predicts context words from target word (better for rare words, large datasets).

**Q: What is attention mechanism and why was it needed?**
> Encoder-decoder RNNs compress entire input into one vector (information bottleneck). Attention lets the decoder look at ALL encoder states and choose which parts to focus on when generating each output word.

**Q: What is the vanishing gradient problem in RNNs?**
> In long sequences, gradients become exponentially small during backpropagation, making it impossible to learn long-range dependencies. Solution: LSTM/GRU gates that control information flow.

### 🔴 Advanced Level

**Q: How does BERT differ from GPT architecturally?**
> BERT: Encoder-only, bidirectional (sees all context simultaneously), pre-trained with MLM+NSP, best for understanding tasks (classification, NER, QA). GPT: Decoder-only, auto-regressive (left-to-right), pre-trained with causal LM, best for generation tasks.

**Q: What is RAG and when would you use it?**
> RAG = Retrieval Augmented Generation. Use when: your LLM needs domain-specific knowledge not in training data, you want to reduce hallucinations, you need up-to-date information, or data privacy prevents fine-tuning.

**Q: Explain the Transformer's attention formula.**
> `Attention(Q,K,V) = softmax(QK^T / √d_k) × V`. Q=Query, K=Key, V=Value. The √d_k scaling prevents vanishing gradients with large dimensions. Multi-head attention runs this h times in parallel with different projections.

**Q: What's the difference between fine-tuning and RAG?**
> Fine-tuning: retrain model weights on domain data (expensive, static). RAG: attach external knowledge base (cheaper, dynamic, updatable). Use fine-tuning for style/behavior, RAG for knowledge.

### 🚀 Expert Level (Agentic AI)

**Q: What is an AI Agent?**
> An AI agent is an LLM that can perceive its environment, reason about goals, plan actions, use tools (web search, calculators, APIs, databases), and take autonomous steps to accomplish complex tasks.

**Q: What is LangGraph and how does it differ from LangChain?**
> LangChain: sequential chains of LLM calls. LangGraph: stateful, cyclical graphs where agents can loop, branch, and coordinate — better for multi-agent workflows where agents can revisit steps.

---

## 📦 Complete Requirements

```python
# requirements.txt for all code in this guide
# pip install -r requirements.txt

nltk==3.8.1
spacy==3.7.2
gensim==4.3.2
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.0.3
tensorflow==2.14.0
torch==2.1.0
transformers==4.36.0
datasets==2.15.0
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.13
faiss-cpu==1.7.4
chromadb==0.4.18
openai==1.6.0
matplotlib==3.7.2
seaborn==0.12.2
```

---

## 🗺️ Learning Roadmap

```
Month 1: Foundations
├── Python basics + NumPy + Pandas
├── Text Preprocessing (Tokenization, Stemming, Lemmatization)
└── Projects: Text cleaner, BOW classifier

Month 2: Classical NLP + ML
├── BOW, TF-IDF, N-grams
├── Scikit-learn ML models (Naive Bayes, SVM, LR)
└── Projects: Spam classifier, Sentiment analysis

Month 3: Word Embeddings
├── Word2Vec (CBOW, Skip-gram)
├── GloVe, FastText
└── Projects: Word analogy finder, Avg Word2Vec sentiment

Month 4: Deep Learning for NLP
├── RNN, LSTM, GRU
├── Attention Mechanism
└── Projects: Sentiment analysis with BiLSTM

Month 5: Transformers & BERT
├── Transformer architecture
├── BERT, RoBERTa, DistilBERT
├── Fine-tuning with Hugging Face
└── Projects: Custom text classifier, NER system

Month 6: LLMs & Gen AI
├── GPT family, Claude, Gemini
├── Prompt engineering
├── RAG systems
└── Projects: Custom chatbot with RAG

Month 7: Agentic AI
├── LangChain agents
├── LangGraph
├── Multi-agent systems
└── Projects: Research agent, Code assistant agent
```

---

*📌 This guide covers: Text Preprocessing → Classical NLP → Word Embeddings → Deep Learning → Transformers → LLMs → Gen AI → Agentic AI*

*🔗 References: Your handwritten notes, NLTK docs, Gensim docs, Hugging Face docs, LangChain docs*
