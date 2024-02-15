from tokenizer import Tokenizer
from Smoothing import GoodTuring, N1byN_v2, calculate_perplexity2, final_smoothed_values, linear_interpolation, calculate_perplexity_linear, estimate_lambda, sum_smooth_for_each_bigram, tokenizerForNgram, tokenizerForNgram_input, generate_ngrams, ngrams_count, count_of_count
import random
import sys
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import numpy as np
from scipy.stats import linregress
import os

#############################PREDICTION FOR GOOD TURING###################################################
def predict_next_word(context, ngram_map, fin_r_to_prob, k, token_set):
    last_two_words = tuple(context.split()[-2:])  
    candidates = {}
  
    for key in ngram_map.keys():
        if key[0] == last_two_words:
            continuation = key[1]
            count = ngram_map[key]
            probability = fin_r_to_prob.get(count, 0)  
            candidates[continuation] = probability

    if len(candidates) < k:
         for token in token_set:
             #new_trigram = last_two_words + (token,)
             f_element = last_two_words
             l_element = token
             new_trigram = (f_element,l_element)
             if new_trigram not in candidates:  
                 if new_trigram in ngram_map:
                     count = ngram_map[new_trigram]
                 else:
                     
                     count = 0  
                 probability = fin_r_to_prob.get(count, 0)  
                 candidates[token] = probability

    sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:k]

    return sorted_candidates
###########################################################################################################################33
###########################FOR THE TYPE TO TOKEN RATIO##################################################################################3
def create_token_set(corpus_tokens):
    token_set = set()
    for token in corpus_tokens:
        token_set.add(token)
    return token_set
##########################################################################################################3

##########################LINEAR INTERPOLATION PREDICTION#####################################################3

def predict_next_word_linear(context, ngram_map,interpolated_prob, k, token_set):
    last_two_words = tuple(context.split()[-2:])  
    #print(last_two_words)
    candidates = {}

    
    for key in ngram_map.keys():
       
        if key[0] == last_two_words:
            continuation = key[1]
            #count = ngram_map[key]
            #print(continuation)
            probability = interpolated_prob.get(key, 0) 
            candidates[continuation] = probability


    if len(candidates) < k:
         for token in token_set:
             f_element = last_two_words
             l_element = token
             new_trigram = (f_element,l_element)
             #print(new_trigram)
             if new_trigram not in candidates:  
              
                 
                 probability = interpolated_prob.get(new_trigram, 1e-3) 
                 candidates[token] = probability


    #selected_candidates = random.sample(list(candidates.items()), k)
    sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:k]

    return sorted_candidates

#################################################################################################################

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        sys.exit(1)

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])

    tokenizer = Tokenizer()

    with open(corpus_path, 'r',encoding='utf-8') as file:
        text = file.read().lower()

   
    output_prefix = ''
    if lm_type == 'g':
        output_prefix = 'goodturing'
    elif lm_type == 'i':
        output_prefix = 'interpolation'

    flattened_list, test_set, train_set = tokenizerForNgram(text)

    corpus_ngrams = generate_ngrams(flattened_list, 3)
    corpus_ngram_map = ngrams_count(corpus_ngrams)
    corpus_count_counts = count_of_count(corpus_ngram_map)
    n1_by_n = N1byN_v2(corpus_count_counts)

    token_set = set()
    token_set = create_token_set(flattened_list)
    


    if lm_type == 'g':
        ngram_estimator = GoodTuring()
        fin_r_to_prob = ngram_estimator.hybrid_estimator(corpus_count_counts, corpus_ngram_map)
        sum_smoothin = sum_smooth_for_each_bigram(fin_r_to_prob,corpus_count_counts,corpus_ngram_map)
        #print(sum_smoothin)
        fin_r_to_prob = final_smoothed_values(corpus_ngram_map,sum_smoothin,fin_r_to_prob)
        fin_r_to_prob[0] = 10**-3
        all_perx_train = []
        sentence_perp_map_train = {}
        for train in train_set:
            perpx_score_train = calculate_perplexity2(train,corpus_ngram_map,corpus_count_counts,3,fin_r_to_prob,n1_by_n)
            all_perx_train.append(perpx_score_train)
            train = train[1:-1]
            sentence = " ".join(train)
            sentence_perp_map_train[sentence] = perpx_score_train


        all_perx_test = []
        sentence_perp_map_test = {}
        for test in test_set:
            perpx_score_test = calculate_perplexity2(test,corpus_ngram_map,corpus_count_counts,3,fin_r_to_prob,n1_by_n)
            all_perx_test.append(perpx_score_test)
            test = test[1:-1]
            sentence = " ".join(test)
            sentence_perp_map_test[sentence] = perpx_score_test

        with open(f"{output_prefix}-{os.path.basename(corpus_path)}-trainingset-perplexity", "w",encoding='utf-8') as file:
            file.write(f"Avg Perplexity: {sum(all_perx_train) / len(all_perx_train)}\n\n\n\n")
            for key, value in sentence_perp_map_train.items():
                file.write(f"{key}: {value}\n\n")
        
        with open(f"{output_prefix}-{os.path.basename(corpus_path)}-testingset-perplexity", "w",encoding='utf-8') as file:
            file.write(f"Avg Perplexity: {sum(all_perx_test) / len(all_perx_test)}\n\n\n\n")
            for key, value in sentence_perp_map_test.items():
                file.write(f"{key}: {value}\n\n")


    elif lm_type == 'i':
        corpus_2grams = generate_ngrams(flattened_list, 2)
        corpus_2gram_map = ngrams_count(corpus_2grams)
        corpus_1grams = generate_ngrams(flattened_list, 1)
        corpus_1gram_map = ngrams_count(corpus_1grams)
        lam1, lam2, lam3 = estimate_lambda(corpus_ngram_map, corpus_2gram_map, corpus_1gram_map)
        interpolated_prob = linear_interpolation(corpus_ngram_map, corpus_2gram_map, corpus_1gram_map, lam1, lam2, lam3)
        all_perx_train = []
        sentence_perp_map_train = {}
        for train in train_set:
            perpx_score_train = calculate_perplexity_linear(train, interpolated_prob)
            all_perx_train.append(perpx_score_train)
            train = train[1:-1]
            sentence = " ".join(train)
            sentence_perp_map_train[sentence] = perpx_score_train

        with open(f"{output_prefix}-{os.path.basename(corpus_path)}-trainingset-perplexity", "w",encoding='utf-8') as file:
            file.write(f"Avg Perplexity: {sum(all_perx_train) / len(all_perx_train)}\n\n\n\n")
            for key, value in sentence_perp_map_train.items():
                file.write(f"{key}: {value}\n\n")

        all_perx_test = []
        sentence_perp_map_test = {}
        for test in test_set:
            perpx_score_test = calculate_perplexity_linear(test, interpolated_prob)
            all_perx_test.append(perpx_score_test)
            test = test[1:-1]
            sentence = " ".join(test)
            sentence_perp_map_test[sentence] = perpx_score_test    

        with open(f"{output_prefix}-{os.path.basename(corpus_path)}-testingset-perplexity", "w",encoding='utf-8') as file:
            file.write(f"Avg Perplexity: {sum(all_perx_test) / len(all_perx_test)}\n\n\n\n")
            for key, value in sentence_perp_map_test.items():
                file.write(f"{key}: {value}\n\n")


    else:
        print("Invalid LM type. Please use 'g' for Good-Turing or 'i' for Interpolation.")
        sys.exit(1)

    sentence_to_check = input("Enter a sentence to calculate perplexity and predict next word: ")
    k = int(input("Enter k (number of candidates for the next word): "))

    split_sentence_to_check = tokenizerForNgram_input(sentence_to_check)


    if lm_type == 'g':
        next_word_predictions = predict_next_word(sentence_to_check, corpus_ngram_map,fin_r_to_prob , k,token_set)
        print("Next word predictions:")
        if len(next_word_predictions)==0:
            print("<UNKNOWN>")
        else:
            for word, prob in next_word_predictions:
               print(f"{word} {prob:.10f}")
    elif lm_type == 'i':
        next_word_predictions_linear = predict_next_word_linear(sentence_to_check, corpus_ngram_map,interpolated_prob, k,token_set)
        print("Next word predictions:")
        if len(next_word_predictions_linear)==0:
            print("<UNKNOWN>")
        else:
            for word, prob in next_word_predictions_linear:
               print(f"{word} {prob:.10f}")
 