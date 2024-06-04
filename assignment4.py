import re
import math
import random

def tokenise(filename):
    with open(filename, 'r') as f:
        return [i for i in re.split(r'(\d|\W)', f.read().replace('_', ' ').lower()) if i and i != ' ' and i != '\n']

def build_unigram(sequence):
    # Task 1.1
    # Return a unigram model.
    # Replace the line below with your code.
    unigram = {}
    for token in sequence: 
        unigram[token] = unigram.get(token,0) + 1
    model = {(): unigram}
    return model


def build_bigram(sequence):
    # Task 1.2
    # Return a bigram model.
    # Replace the line below with your code.
    bigram = {}
    for i in range(len(sequence) - 1): 
        key = (sequence[i],)
        token = sequence[i+1]
        if key not in bigram:
            bigram[key] = {token: 1}
        else:
            bigram[key][token] = bigram[key].get(token, 0) + 1
    return bigram


def build_n_gram(sequence, n):
    # Task 1.3
    # Return an n-gram model.
    # Replace the line below with your code.
    ngram = {}
    for i in range(len(sequence) -(n - 1)): 
        key = []
        j = 0
        while j < n:
            key.append(sequence[i + j - 1])
            j += 1
        token = sequence[i + (n - 1)]
        key = tuple(key)
        if token not in ngram:
            ngram[key] = {token, ngram.get(token, 0) + 1}
    return ngram

def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    raise NotImplementedError

def blend_predictions(preds, factor=0.8):
    # Task 3
    # Return a blended prediction as a dictionary.
    # Replace the line below with your code.
    raise NotImplementedError

def predict(sequence, models):
    # Task 4
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_ramp_up(sequence, models):
    # Task 5.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_blended(sequence, models):
    # Task 5.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

if __name__ == '__main__':

    sequence = tokenise('assignment4corpus.txt')

    # Task 1.1 test code
    '''
    model = build_unigram(sequence[:20])
    print(model)
    '''

    # Task 1.2 test code
    '''
    model = build_bigram(sequence[:20])
    print(model)
    '''

    # Task 1.3 test code
    
    model = build_n_gram(sequence[:20], 5)
    print(model)
    
    
    # Task 2 test code
    '''
    print(query_n_gram(model, tuple(sequence[:4])))
    '''

    # Task 3 test code
    '''
    other_model = build_n_gram(sequence[:20], 1)
    print(blend_predictions([query_n_gram(model, tuple(sequence[:4])), query_n_gram(other_model, ())]))
    '''

    # Task 4 test code
    '''
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = predict(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    '''

    # Task 5.1 test code
    '''
    print(log_likelihood_ramp_up(sequence[:20], models))
    '''

    # Task 5.2 test code
    '''
    print(log_likelihood_blended(sequence[:20], models))
    '''

