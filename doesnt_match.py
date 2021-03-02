# doesnt_match("breakfast cereal dinner lunch")
# 'cereal'

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
