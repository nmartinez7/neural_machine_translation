#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:27:10 2022

@author: s2268276
"""

#%%
import sys

#%%
# Given a file, return a dictionary of words and freq
def word_frequency(fp):
    data = fp.read()
    words = data.split()
    #print(words)
    fp.close()
   
    total_word_count = 0
    wordfreq = {}
    for word in words:
        total_word_count += 1
        if word not in wordfreq:
            wordfreq[word] = 0 
        wordfreq[word] += 1
        
    print("The total number of words is ", total_word_count)
    #print(wordfreq)
    return wordfreq
    
# Given a dictionary of word and frequency, returns the words
# with frequency 1, sorted alphabetically
def words_with_freq_1(wordfreq):
    wordfreq_1 = []
    for word in wordfreq:
        if(wordfreq[word] == 1):
            wordfreq_1.append(word)
    wordfreq_1.sort()
    return wordfreq_1
    
#%%
#GERMAN
fp_german = open('/afs/inf.ed.ac.uk/user/s22/s2268276/nlu-cw2/europarl_raw/train.de', "r")
wordfreq_german = word_frequency(fp_german)
wordfreq_1_german = words_with_freq_1(wordfreq_german)
#print(wordfreq_1_german)
print("Number of tokens in German", len(wordfreq_german))
print("Number of tokens replaced in German", len(wordfreq_1_german))

#%%
#ENGLISH
fp_english = open('/afs/inf.ed.ac.uk/user/s22/s2268276/nlu-cw2/europarl_raw/train.en', "r")
wordfreq_english = word_frequency(fp_english)
wordfreq_1_english = words_with_freq_1(wordfreq_english)
#print(wordfreq_1_english)
print("Number of tokens in English", len(wordfreq_english))
print("Number of tokens replaced in English", len(wordfreq_1_english))


#%%
#QUESTION 3.4 - Same words in English and German
wordfreq_1_inboth = set(wordfreq_1_english)&set(wordfreq_1_german)
print("Number of tokens replaced that are the same in English and German", len(wordfreq_1_inboth))
#print(wordfreq_1_inboth)