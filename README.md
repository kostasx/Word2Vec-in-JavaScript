## Word2Vec with TensorFlow.JS

![](screenshot.jpg)

## [Demo](https://kostasx.github.io/Word2Vec-in-JavaScript/)

### [What are Word Embeddings](https://www.wikiwand.com/en/Word_embedding)

### Todo

- Implement function `getAntonym()`. References:
    - [How to obtain antonyms through Word2Vec](https://stackoverflow.com/questions/31814825/how-to-obtain-antonyms-through-word2vec)
    - [Exploring Antonyms](https://gist.github.com/kostasx/cb40e695588370faafd70b78b6c1f773)
- Implement function `doesnt_match` from Python:

Which word from the given list doesn't go with the others?

```py
doesnt_match("breakfast cereal dinner lunch")
# 'cereal'
```

Python source:

```py
def doesnt_match(self, words):

    self.init_sims()
    words = [word for word in words if word in self.vocab]  # filter out OOV words

    if not words:
        raise ValueError("cannot select a word from an empty list")

    vectors = vstack(self.syn0norm[self.vocab[word].index] for word in words).astype(REAL)

    # 1) take the mean of all the word-vectors – a sort of 'center' for all candidates
    mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)

    # 2) calculate the cosine-distance from that center to each word – this is the dot-product between unit-normalized versions of each relevant vector
    dists = dot(vectors, mean)

    # 3) return the single word with the highest cosine-distance from that mean
    return sorted(zip(dists, words))[0][1]
```

