import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import numpy as np
from scipy.stats import linregress
import random

class Tokenizer:

    def __init__(self):
       pass

    def replace_contractions(self,text):
    
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "I'd": "I would",
            "I'd've": "I would have",
            "I'll": "I will",
            "I'll've": "I will have",
            "I'm": "I am",
            "I've": "I have",
            "isn't": "is not",
            "it'd": "it had",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there had",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'alls": "you alls",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have"
        } 
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)

        return text   

    def replace_salutations(self, text):
        salutations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Fr.']
        for salutation in salutations:
            text = re.sub(r'\b' + re.escape(salutation) + r'\b', '<Salutation>', text)
        return text
    
    def replace_punctuation1(self,sentence_tokens):
     
        punctuation_pattern = re.compile(r'[^\w\s<>]+')
        flat_tokens = [token for sentence in sentence_tokens for token in sentence]
        replaced_tokens = [punctuation_pattern.sub('<PUNC>', token) for token in flat_tokens]
        replaced_tokens = [token for token in replaced_tokens if token != '<PUNC>']
        return replaced_tokens
    
    
    def replace_punctuation(self, sentence_tokens):
        punctuation_pattern = re.compile(r'[^\w\s<>]+')
        replaced_tokens = []
        for sentence in sentence_tokens:
            replaced_sentence = [punctuation_pattern.sub('<PUNC>', token) for token in sentence]
            replaced_tokens.append(replaced_sentence)
        new_sentence_tokens = [[token for token in sentence if token != '<PUNC>'] for sentence in replaced_tokens]
        return new_sentence_tokens

    
    
    def custom_sentence_splitter(self,text):
        sentences = []
        current_sentence = ""
        sentence_endings = {'.', '?', '!'}
        for char in text:
            current_sentence += char
            if char in sentence_endings:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence:
            sentences.append(current_sentence.strip())
        return sentences

    def replace_entities(self, text):       
        text = self.replace_contractions(text)
        text = re.sub(r'#\w+', '<HASHTAG>', text)
        text = re.sub(r'(https?://|www\.)[\w.-]+(?:\.[\w.-]+)+[\w\-.?=%&=\+#/]*', '<URL>', text)
        text = re.sub(r'@\S+', '<MENTION>', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<MAILID>', text)
        text = re.sub(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', '<DATE>', text)
        text = re.sub(r'\d{1,2}:\d{1,2}(:\d{1,2})?', '<TIME>', text)
        text = re.sub(r'\b(?:\+\d{1,4}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b', '<PHONENO>', text)
        text = re.sub(r'\b\d+\b', '<NUM>', text)
        text = self.replace_salutations(text)    
        sentences = self.custom_sentence_splitter(text)
        sentences = list(filter(str.strip, sentences))
        return sentences

    def replace_entities2(self, text):
        text = self.replace_entities(text)
        sentences_with_sos_eos = ['<SOS>' + ''.join(sentence) + '<EOS>' for sentence in text]
        word_tokenized_sentences = [re.findall(r'[^\w\s<>]+|\b\w+\b|<\w+>', sentence) for sentence in sentences_with_sos_eos]
        word_tokenized_sentences = self.replace_punctuation(word_tokenized_sentences)        
        return word_tokenized_sentences

    def replace_entities3(self, text):
        text = self.replace_entities(text)
        word_tokenized_sentences = [re.findall(r'[^\w\s<>]+|\b\w+\b|<\w+>', sentence) for sentence in text]     
        return word_tokenized_sentences


def main():   
    tokenizer = Tokenizer()  
    user_input = input("Type any text: ")  
    tokenized_result = tokenizer.replace_entities3(user_input)   
    print()
    print()
    print("Tokenized Result:", tokenized_result)

if __name__ == "__main__":
    main()
