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
    # loop through sequence 
    for i in range(len(sequence) - (n - 1)): 
        #create key with n-1 tokens in it
        key = tuple(sequence[i:i + n - 1])
        #get the correct token
        token = sequence[i + (n - 1)]
        #init dict key if not already 
        if key not in ngram:         
            ngram[key] = {}
        #update count for token
        ngram[key][token] = ngram[key].get(token, 0) + 1
    return ngram

def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    #if sequence empty, goes unigram
    for token in model: 
        len_token = len(token)
        break
    if len_token == 0:  
        return model.get(())
    elif len_token >= 1:
        if sequence in model: 
            return model[sequence]
    return None


def blend_predictions(preds, factor=0.8):
    # Task 3
    # Return a blended prediction as a dictionary.
    # Replace the line below with your code.

    from collections import Counter

    #remove NONE values from preds
    preds = [p for p in preds if p is not None]

    #normilize each pred to sum to 1 
    norm_preds = []
    for pred in preds: 
        factor = 1.0/sum( pred.values())
        for k in pred:
            pred[k] = pred[k]*factor 

        #if total is larger then 1 normilze 
        
    #Blend according to the factor 
    #init blended and weights 
    blended = {}
    col_list = []


    #loop through each pred and calc weight
    for i in range(len(preds)):
        if i == 0: 
            weight = 0.8
        else: 
            weight = math.pow((1.0 - 0.8),(i))
        
        for token in preds[i]:
            preds[i][token] *= weight 


    # add to the list 
    for i in range(len(preds)):
        col_list.append(Counter(preds[i]))
    blended = dict(sum(col_list, Counter()))
    return blended



def predict(sequence, models):
    # Task 4
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.
    all_pred = []
    
    #check context sequence is sufficient for the n gram 
    len_seq = len(sequence)
    for model in models:
        for token in model: 
            n_value = len(token)
            break 
        if len_seq >= n_value:
            all_pred.append(query_n_gram(model, tuple(sequence[-n_value:])))

    #blend thme together 
    blend_pred= blend_predictions(all_pred)
    preds = random.choices(list(blend_pred.keys()), weights=[blend_pred[i] for i in blend_pred], k =1 )[0]
    return preds
        


    

    




def log_likelihood_ramp_up(sequence, models):
    # Task 5.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    log_likelihood = 0 
    for i in range(len(sequence)):
        token = sequence[i]
        #for unigrams the first loop through should have no context 
        if i == 0:
            context = ()
        else: 
            #gets correct length of n gram for other tokens
            context = tuple(sequence[max(0, i - (len(models)- 1)):i])
        model_index = min(len(context), len(models) - 1)
        
        model = models[model_index]
        #print(model)
        #query model with the correct context 
        pred = query_n_gram(model, context)

        #return -math.inf if doesn't exist 
        if pred is None or token not in pred:
            return -math.inf

        prob = pred[token] / sum(pred.values())
        print(prob)

        log_likelihood = math.log(prob)
  
    return log_likelihood




def log_likelihood_blended(sequence, models):

    # Task 5.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

if __name__ == '__main__':

    sequence = tokenise('assignment4corpus.txt')

    # Task 1.1 test code
    
    model = build_unigram(sequence[:20])
    print(model)
    

    # Task 1.2 test code
    
    model = build_bigram(sequence[:20])
    print(model)
    

    # Task 1.3 test code
   
    model = build_n_gram(sequence[:20], 5)
    print(model)
    

    # Task 2 test code
    
    print(query_n_gram(model, tuple(sequence[:4])))
    

    # Task 3 test code
   
    other_model = build_n_gram(sequence[:20], 1)
    print(blend_predictions([query_n_gram(model, tuple(sequence[:4])), query_n_gram(other_model, ())]))
    

    # Task 4 test code

    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = predict(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()


    # Task 5.1 test code
    
    print(log_likelihood_ramp_up(sequence[:20], models))
    

    # Task 5.2 test code
    '''
    print(log_likelihood_blended(sequence[:20], models))
    '''

