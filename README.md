# NLP_NextWordPredictor

A trigram next-word predictor built in Python for the NLP Systems course at Moraine Park Technical College (Assignment 7.5).

The model uses the previous two words to predict the next word, then chains those predictions together to generate text.

## How It Works

1. **Preprocess:** lowercase the text, remove punctuation, and split it into tokens. Each sentence gets start (`<s>`) and end (`</s>`) markers so sentences stay separate.
2. **Count trigrams:** loop through the tokens and count which word follows each word pair, e.g. `('i', 'love') -> {'natural': 2, 'machine': 1, ...}`.
3. **Compute probabilities:** divide each count by the total for that pair, e.g. `('i', 'love') -> {'natural': 0.4, 'machine': 0.2, ...}`.
4. **Predict:** given two words, return the top 3 most likely next words.
5. **Generate:** start with two words and repeatedly add the most likely next word until the sentence ends.

## Example Output

```
Context: 'i love'
1. natural (0.400)
2. machine (0.200)
3. learning (0.200)

Prompt: 'python is'
Generated: python is great for machine learning
```

## How to Run

Requires Python 3 (no extra packages needed).

```
python next_word_predictor.py
```

## Limitations

A trigram model only sees two words of context, so it can't follow the meaning of a full sentence. It also can't predict anything for word pairs it never saw during training, and it has no sense of which words are similar in meaning. Modern language models get around this by using much longer context and learned word embeddings.
