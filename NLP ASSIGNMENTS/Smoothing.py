import random
import sys
from tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import numpy as np
from scipy.stats import linregress
import os

class GoodTuring:
    def __init__(self):
        self.turing_estimates = {} 
        self.turing_variance = {}

    def calculate_n1_by_n(self, count_counts):
        #total_events = sum(count_counts.values())
        N = sum(k * count_counts[k] for k in count_counts)
        n1_by_n = count_counts[1] / N
        return n1_by_n


    def linear_good_turing_estimator(self, count_counts):
        log_r_values, log_Zr_values = self.calculate_log_values(count_counts)
        slope, intercept = self.calculate_regression(log_r_values, log_Zr_values)
        r_star_LGT = {}
        for r in count_counts:
            numerator = np.exp(intercept + slope * np.log(r+1)) * (r+1)
            denominator = np.exp(intercept + slope * np.log(r))
            r_star_LGT[r] = numerator / denominator
        return r_star_LGT

    def calculate_log_values(self, count_counts):
        log_rank_values = []
        log_smoothed_values = []
        sorted_counts = sorted(count_counts.keys())

        for index_of_count, count in enumerate(sorted_counts):
            if count_counts[count] > 0:
                if index_of_count > 0:
                    previous_count = sorted_counts[index_of_count - 1]
                else:
                    previous_count = 0

                if index_of_count < len(sorted_counts) - 1:
                    next_count = sorted_counts[index_of_count + 1]
                else:
                    next_count = count

                if count != sorted_counts[-1]:
                    numerator = count_counts[count]
                    denominator = 0.5 * (next_count - previous_count)
                else:
                    numerator = count_counts[count]
                    denominator = count - previous_count

                smoothed_value = numerator / denominator
                log_rank_values.append(np.log(count))
                log_smoothed_values.append(np.log(smoothed_value))

        return log_rank_values, log_smoothed_values



    def calculate_regression(self, log_rank_values, log_smoothed_values):
        try:
            slope, intercept, _, _, _ = linregress(log_rank_values, log_smoothed_values)
        except ValueError:
            
            slope, intercept = 0, 0

        return slope, intercept

    def N1byN(self, count_counts):
        N = sum(val*count_counts[val] for val in count_counts)
        N1_by_N = count_counts.get(1, 0) / N
        return N1_by_N

    

    def hybrid_estimator(self, count_counts, ngram_map):
        turing_estimates={}
        turing_variance={}
        total_counter_for_normalisation = sum(ngram_map.values())
        N1_by_N = self.N1byN(count_counts)
        turing_estimates,turing_variance = self.final_turing(count_counts,turing_estimates,turing_variance)
        lgt_estimates = self.linear_good_turing_estimator(count_counts)
        chosen_estimates = self.toggle_between_lgt_and_fin(turing_estimates, lgt_estimates, count_counts,turing_variance)
        return chosen_estimates

    def toggle_between_lgt_and_fin(self, turing_estimates, lgt_estimates, count_counts, turing_variance):
        chosen_estimates = {}
        lgt_flag = False

        for count in sorted(count_counts):
            condition_1 = lgt_flag
            condition_2 = count not in turing_estimates

            if condition_1 or condition_2:
                chosen_estimates[count] = lgt_estimates[count]
            else:
                difference = abs(turing_estimates[count] - lgt_estimates[count])
                threshold = 1.65 * np.sqrt(turing_variance[count])

                if difference <= threshold:
                    chosen_estimates[count] = turing_estimates[count]
                else:
                    chosen_estimates[count] = lgt_estimates[count]
                    lgt_flag = True

            if count == 0:
                print(chosen_estimates[count])
            # print("CHOSEN ESTIMATES")        
        #print(chosen_estimates)
        return chosen_estimates

    
    def calculate_unnormalized_probabilities(self, chosen_estimates, total_counter_for_normalisation):
        unnormalized_probabilities = {}

        for count, estimate in chosen_estimates.items():
            probability = estimate / total_counter_for_normalisation
            unnormalized_probabilities[count] = probability

        return unnormalized_probabilities


    def fin_normalize_probabilities(self, unnormalized_probabilities, N1_by_N):
        probabilities = {}

        sum_unnormalized_probabilities = sum(unnormalized_probabilities.values())
        
        for count, p in unnormalized_probabilities.items():
            normalized_probability = (1-N1_by_N)*(p / sum_unnormalized_probabilities)
            probabilities[count] = normalized_probability
        
        probabilities[0] = N1_by_N

        return probabilities



    def final_turing(self,count_counts,turing_estimates,turing_variance):

        for c, Nc in count_counts.items():
            Nc_plus1 = count_counts.get(c + 1, 0)

            if Nc_plus1 > 0:
                turing_estimates[c] = (c + 1) * Nc_plus1 / Nc
                turing_variance[c] = ((c + 1)**2) * Nc_plus1 / (Nc**2) * (1 + (Nc_plus1 / Nc))
            
            else:
                turing_estimates[c] = c
                turing_variance[c] = 0

        return turing_estimates, turing_variance
 ######################################################33   
def N1byN_v2(count_counts):
        N = sum(val*count_counts[val] for val in count_counts)
        N1_by_N = count_counts.get(1, 0) / N
        return N1_by_N
#############################################################
def tokenizerForNgram(text):   
    tokenizer = Tokenizer()
    tokenized_sentences = tokenizer.replace_entities2(text)
   
    train_set, test_set = train_test_split(tokenized_sentences, test_size=0.125,random_state=42)
    one_d_list = [item for sublist in train_set for item in sublist]
    return one_d_list,test_set,train_set

######################TRAIN -TEST -SPLIT AND TOKENIZE CORPUS###################################
def ngram_tok(text,tok):
    word_tokenized_sentences = tok.replace_entities2(text)
    train_sentences, test_sentences = train_test_split(word_tokenized_sentences, test_size=1000, random_state=42)
    train_flattened = [token for sentence in train_sentences for token in sentence]
    return train_flattened, test_sentences,train_sentences
###############################GENERATE N-GRAMS################################################################

def generate_ngrams(tokens, n):
    ngrams=[]
    tokens = (n-1)*['<SOS>']+tokens

    for i in range(n-1, len(tokens)):
       prefix = tuple(tokens[i -p-1] for p in reversed(range(n-1)))
       target = tokens[i]
       ngrams.append((prefix, target))
    return ngrams
#############################################################################################################



def generate_ngrams1(tokens, n):
    ngrams = []
    
    modified_tokens = []

    for token in tokens:
        if token == '<SOS>':
            modified_tokens.extend(['<SOS>', '<SOS>'])
        else:
            modified_tokens.append(token)

    for i in range(n - 1, len(modified_tokens)):
        prefix = tuple(modified_tokens[i - p - 1] for p in reversed(range(n - 1)))
        target = modified_tokens[i]
        ngrams.append((prefix, target))
    return ngrams



#############################N-GRAM COUNTER##################################################################
def ngrams_count(ngrams):
    ngram_map={}
    for ind_tuple in ngrams:
       if ind_tuple in ngram_map:
          ngram_map[ind_tuple]+=1
       else:
          ngram_map[ind_tuple]=1
    #print("ngrams_count",ngram_map)
    return ngram_map
##############################################################################################################

##########################COUNT OF COUNT#########################################################
def count_of_count(ngram_map):
    count_counts = defaultdict(int)
    for count in ngram_map.values():
        count_counts[count] += 1
    #print("count_counts",count_counts)
    return count_counts

###########################FOR THE TYPE TO TOKEN RATIO##################################################################################3
def create_token_set(corpus_tokens):
    token_set = set()
    for token in corpus_tokens:
        token_set.add(token)
    return token_set
#########################

#########################PERPLEXITY DUPLICATE############################################################333
def calculate_perplexity2(tokens, ngram_map, count_counts, n,fin_r_to_prob,n1_by_n):
    ngrams = generate_ngrams(tokens, n)
    log_perplexity_sum = 0.0

    for ngram in ngrams:
        count = ngram_map.get(ngram, 0)
        probability = fin_r_to_prob[count]
        if probability > 0:  
            log_probability = math.log2(probability)
            log_perplexity_sum += log_probability
        else:
         
            log_probability = n1_by_n
            log_perplexity_sum += log_probability

    avg_log_perplexity = log_perplexity_sum / len(ngrams)
    perplexity = 2 ** (-avg_log_perplexity)
    return perplexity
#########################################################################################################################
################################PROBABILITY OF A SENTENCE TO BE PRINTED ON SCREEN########################################
'''
def calculate_probability(tokens, ngram_map, count_counts, n, fin_r_to_log_prob):
    ngrams = generate_ngrams(tokens, n)
    print("N-grams:", ngrams)
    log_probability_sum = 0.0
    probability_product = 1.0
    for ngram in ngrams:
        count = ngram_map.get(ngram, 0)
      #  if count != 0:
       #     print(ngram, "MATCH")
        #else:
         #   print(ngram, "Mismatch")

        probability = fin_r_to_prob[count]
        log_probability = math.log2(probability)
        #probability = fin_r_to_prob[count]
        # print("Probability:", probability)
        log_probability_sum += log_probability
        #probability_product *= probability

    final_probability = 2 ** log_probability_sum
    print(final_probability)
    return final_probability
 '''
##############################PERPLEX CHECK FUNCTION#####################################################
def tokenizerForNgram_input(text):
    tokenizer = Tokenizer()
    tokenized_sentences = tokenizer.replace_entities3(text)
    one_d_list = [item for sublist in tokenized_sentences for item in sublist]
    return one_d_list
##########################################################################################################


##################################LAMBDA ESTIMATION###############################################################

def estimate_lambda(corpus_ngram_map, corpus_2gram_map, corpus_1gram_map):
    lambda1 = 0
    lambda2 = 0
    lambda3 = 0


    for trigram, trigram_count in corpus_ngram_map.items():
      t2 = ((trigram[0][1],),trigram[1])
      #print(trigram[0])
      f_of_t2_t3 = corpus_2gram_map.get(t2, 0)
      t3 = ((trigram[0][0],),trigram[0][1])
      f_of_t1_t2 = corpus_2gram_map.get(t3, 0)
      f_of_t1_t2_t3 = corpus_ngram_map.get(trigram,0)
      second_last_element_of_trigram = trigram[0][1]
      last_element_of_trigram = trigram[1]
      converted_format1 = ((), second_last_element_of_trigram)
      #print(converted_format1)
      converted_format2 = ((), last_element_of_trigram)
      f_of_t2 = corpus_1gram_map.get(converted_format1, 0)
      f_of_t3 = corpus_1gram_map.get(converted_format2, 0)
      the_N = sum(corpus_1gram_map.values())
      temp=0
      if f_of_t1_t2_t3>0:

          if f_of_t1_t2 > 1:
             case1 = (f_of_t1_t2_t3 - 1) / (f_of_t1_t2 - 1)
          else:
             case1 = 0
          
          if f_of_t2> 1:
             case2 = (f_of_t2_t3 - 1) / (f_of_t2 - 1)
          else:
             case2 = 0
           
          case3 = (f_of_t3 - 1) / (the_N - 1)
        
          temp = max(case1,case2,case3)

          if temp == case1 :
            lambda3+=f_of_t1_t2_t3
          elif temp == case2:
            lambda2+=f_of_t1_t2_t3
          else:
            lambda1+=f_of_t1_t2_t3



       
    total_lambda = lambda1+lambda2+lambda3
    #print(lambda1)
    #print(lambda2)
    #print(lambda3)

    if total_lambda == 0:
        lambda1 = 0
        lambda2=  0
        lambda3  = 0
       # print("lambda1:",lambda1)
       # print("lambda2:",lambda2)
        #print("lambda3:",lambda3)
    else:
        lambda1 /= total_lambda
        lambda2 /= total_lambda
        lambda3 /= total_lambda
       # print("lambda1:",lambda1)
       # print("lambda2:",lambda2)
       # print("lambda3:",lambda3)

    return lambda1, lambda2, lambda3
################LINEAR INTERPOLATION CALCULATION###########################
def linear_interpolation(corpus_ngram_map,corpus_2gram_map,corpus_1gram_map, lambda1, lambda2, lambda3):
     interpolated_probabilities = {}
     for trigram, trigram_count in corpus_ngram_map.items():
        t2 = ((trigram[0][1],),trigram[1])
        f_of_t2_t3 = corpus_2gram_map.get(t2, 0)
        t3 = ((trigram[0][0],),trigram[0][1])
        f_of_t1_t2 = corpus_2gram_map.get(t3, 0)
        f_of_t1_t2_t3 = corpus_ngram_map.get(trigram,0)
        second_last_element_of_trigram = trigram[0][1]
        last_element_of_trigram = trigram[1]
        converted_format1 = ((), second_last_element_of_trigram)
        converted_format2 = ((), last_element_of_trigram)
        f_of_t2 = corpus_1gram_map.get(converted_format1, 0)
        f_of_t3 = corpus_1gram_map.get(converted_format2, 0)
        the_N = sum(corpus_1gram_map.values())
        f_of_t2 = f_of_t2 if f_of_t2 > 0 else 0
        f_of_t1_t2 = f_of_t1_t2 if f_of_t1_t2 > 0 else 0

        part1 = lambda1*(f_of_t3 / the_N)
        if f_of_t2!=0:
          part2 = lambda2* (f_of_t2_t3 / f_of_t2)
        else:
          part2 = 0
        if f_of_t1_t2!=0:
          part3 = lambda3* (f_of_t1_t2_t3 / f_of_t1_t2)
        else:
          part3 = 0


        interpolation_prob = part1+part2+part3
        

        interpolated_probabilities[trigram] = interpolation_prob

    # print(interpolated_probabilities)
     return interpolated_probabilities

   
##############################################################################################
################################printing helper####################################################
def print_interpolated_results(interpolated_probabilities, num_trigrams=25):
    count = 0
    for trigram, probability in interpolated_probabilities.items():
       # print(f"Trigram: {trigram}, Interpolated Probability: {probability}")
        count += 1
        if count == num_trigrams:
            break

#########################################################################33
#############################calculate linear perplexity#############################################33
def calculate_perplexity_linear(sentence, interpolated_prob):
    ngrams = generate_ngrams(sentence, 3)
   

    log_perplexity = 0.0
    for trigram in ngrams:
        probability = interpolated_prob.get(trigram, 1e-3)
        log_perplexity += math.log2(probability)

    avg_log_perplexity = (log_perplexity / len(ngrams))*-1
    perplexity = math.pow(2, avg_log_perplexity)
    return perplexity

#######################################################################################################
################################BIGRAM TO SMOOTHED VALUES##############################################3
def sum_smooth_for_each_bigram(fin_r_to_prob, corpus_count_counts, corpus_ngram_map):
    sum_smoothed_values = {}
    
    for bigram, bigram_count in corpus_ngram_map.items():
        key_for_search = bigram[0]
        #print(key_for_search)
        
        try:
            sum_smoothed_values[key_for_search] += fin_r_to_prob[bigram_count]
        except KeyError:
            sum_smoothed_values[key_for_search] = fin_r_to_prob[bigram_count]

        #print(fin_r_to_prob[bigram_count])

    return sum_smoothed_values
###############################################################################################################
def final_smoothed_values(corpus_ngram_map,sum_smoothin,fin_r_to_prob):
    sum_smoothed_values2 = {}
    
    for bigram, bigram_count in corpus_ngram_map.items():
        key_for_search = bigram[0]
        temp1 = fin_r_to_prob[bigram_count]
        temp2 = sum_smoothin[key_for_search]
        try:
            sum_smoothed_values2[bigram_count]  = temp1/temp2
        except KeyError:
            sum_smoothed_values2[bigram_count]  = temp1/temp2

    return sum_smoothed_values2

############################################DRIVER######################################################################3

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 smoothing.py <lm_type> <corpus_path>")
        sys.exit(1)

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]

    tokenizer = Tokenizer()

    with open(corpus_path, 'r',encoding='utf-8') as file:
        text = file.read().lower()
   
    output_prefix = ''
    if lm_type == 'g':
        output_prefix = 'goodturing'
    elif lm_type == 'i':
        output_prefix = 'interpolation'

    flattened_list, test_set, train_set = tokenizerForNgram(text)
       
    corpus_ngrams = generate_ngrams1(flattened_list, 3)
    
   
    corpus_ngram_map = ngrams_count(corpus_ngrams)
    #print(corpus_ngram_map)l
    corpus_count_counts = count_of_count(corpus_ngram_map)
    n1_by_n = N1byN_v2(corpus_count_counts)

    if lm_type == 'g':
        ngram_estimator = GoodTuring()
        fin_r_to_prob = ngram_estimator.hybrid_estimator(corpus_count_counts, corpus_ngram_map)
        #print(fin_r_to_prob)
        sum_smoothin = sum_smooth_for_each_bigram(fin_r_to_prob,corpus_count_counts,corpus_ngram_map)
        #print(sum_smoothin)
        fin_r_to_prob = final_smoothed_values(corpus_ngram_map,sum_smoothin,fin_r_to_prob)
        fin_r_to_prob[0] = 10**-3
        #print(fin_r_to_prob)
        all_perx_train = []
        sentence_perp_map_train = {}
        for train in train_set:
          
          #if len(train) >3:  
            perpx_score_train = calculate_perplexity2(train,corpus_ngram_map,corpus_count_counts,3,fin_r_to_prob,n1_by_n)
            all_perx_train.append(perpx_score_train)
            train = train[1:-1]
            sentence = " ".join(train)
            sentence_perp_map_train[sentence] = perpx_score_train


        all_perx_test = []
        sentence_perp_map_test = {}      
        fin_r_to_prob[0]=10**-3       
        for test in test_set:
          #if len(test) >3:   
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
       # print(corpus_2gram_map)
        corpus_1grams = generate_ngrams(flattened_list, 1)
        corpus_1gram_map = ngrams_count(corpus_1grams)
       # print(corpus_1gram_map)
        lam1, lam2, lam3 = estimate_lambda(corpus_ngram_map, corpus_2gram_map, corpus_1gram_map)
        interpolated_prob = linear_interpolation(corpus_ngram_map, corpus_2gram_map, corpus_1gram_map, lam1, lam2, lam3)
        all_perx_train = []
        sentence_perp_map_train = {}
        for train in train_set:
          #if len(train) >3:  
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
         # if len(train) >3:
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

  
    while True:
        user_input = input("Enter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        split_sentence_to_check = tokenizerForNgram_input(user_input)

        if lm_type == 'g':
            perplexity_score = calculate_perplexity2(split_sentence_to_check, corpus_ngram_map, corpus_count_counts, 3, fin_r_to_prob,n1_by_n)
           # log_prob = calculate_probability(train,corpus_ngram_map,corpus_count_counts,3,fin_r_to_prob)
            print(f"Perplexity for '{user_input}': {perplexity_score:.4f}")
           # print(f"log for '{user_input}': {log_prob:.4f}")
        elif lm_type == 'i':
            perplexity_score = calculate_perplexity_linear(split_sentence_to_check, interpolated_prob)
            print(f"Perplexity for '{user_input}': {perplexity_score:.4f}")

    print("Exiting program.")