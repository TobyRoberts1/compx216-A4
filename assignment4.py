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
    from collections import Counter

    # Remove NONE values from preds
    preds = [p for p in preds if p is not None]

    # Normalize each pred to sum to 1 
    for pred in preds: 
        total = sum(pred.values())
        if total > 0:
            normalization_factor = 1.0 / total
            for k in pred:
                pred[k] *= normalization_factor 

    # Initialize blended predictions and remaining weight
    blended = {}
    remaining_weight = 1.0

    # Loop through each normalized pred and calculate weight
    for i in range(len(preds)):
        if i == len(preds) - 1:
            # Last prediction takes all remaining weight
            weight = remaining_weight
        else:
            weight = factor * remaining_weight
            remaining_weight -= weight

        for token in preds[i]:
            if token not in blended:
                blended[token] = 0
            blended[token] += preds[i][token] * weight

    # Normalize blended predictions to sum to 1
    total_blended = sum(blended.values())
    if total_blended > 0:
        blended = {k: v / total_blended for k, v in blended.items()}

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
    len_models = len(models) - 1
    model = {}
    for i in range(len(sequence)):
        token = sequence[i ]
        #for unigrams the first loop through should have no context 
        if i == 0:
            context = ()
        else: 
            #gets correct length of n gram for the rest that do have context 
            context = tuple(sequence[max(0, i - len_models):i])
        model_index = min(len(context), len_models)
   
        model = models[len_models -model_index]
   
        #query model with the correct context  
        pred = query_n_gram(model, context)
        
        
        #return -math.inf if doesn't exist 
        if pred is None or token not in pred:
            return -math.inf
        
        prob = pred[token] / sum(pred.values())
       

        log_likelihood += math.log(prob)
    #i beleive in this it is slightly rounded or possible just wrong as it returns about 0.0000000000000056 smaller then meant to be. 
    #I think this is from in task 3 instead of summing to 1 it sums to 1.00000000000000002
    return log_likelihood



def log_likelihood_blended(sequence, models):
    # Task 5.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    blended_log_likelihood = 0 
    len_models = len(models) - 1
    model = {}
    for i in range(len(sequence)):
        token = sequence[i]
        #for unigrams the first loop through should have no context 
        if i == 0:
            context = ()
        else: 
            #gets correct length of n gram for the rest that do have context 
            context = tuple(sequence[max(0, i - len_models):i])

        all_preds = []
        for model in models:
            #query model with the correct context  
            pred = query_n_gram(model, context)
            all_preds.append(pred)
        
        blended_pred = blend_predictions(all_preds)
        

        #return -math.inf if doesn't exist  
        if pred is None or token not in pred:
            return -math.inf

        # calculates the probability of the token
        prob = blended_pred[token]
        #add the log of prob to get the log_likelihood 
        blended_log_likelihood += math.log(prob)
  
    return blended_log_likelihood
    



















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
    
    print(log_likelihood_blended(sequence[:20], models))
    
