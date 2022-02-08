#!/usr/bin/env python
import nltk
import os
from nltk.corpus import brown as corpus
import numpy as np
import pandas as pd
import logging
from argparse import ArgumentParser
from tqdm import tqdm


class HMMLearn:
    logger = logging.getLogger(__name__)
    
    def __init__(self, training_file, model_dump_path):
        self.training_file = training_file
        self.tags_sequence_ls = []
        self.tag_word_tupl_ls = []
        self.unique_tags = {"START", "END"} #Without START and END state
        self.unique_words = set()
        self.model_dump_path = model_dump_path

    def _download_brown_corpus(self):
        self.logger.info("Downloading NLTK 'brown' Corpus and 'universal-tagset' for POS Tagging")
        nltk.download('brown')
        nltk.download('universal_tagset')

    def _eval_emission_probabilities(self):
        self.logger.info("Calculating Emission Probabilities...")
        unique_words_ls=list(self.unique_words)
        len_unique_words = len(self.unique_words)
        unique_tags_ls = list(self.unique_tags)
        len_unique_tags = len(self.unique_tags)
        
        cfd_tag_words= nltk.ConditionalFreqDist(self.tag_word_tupl_ls)
        cpd_tag_words = nltk.ConditionalProbDist(cfd_tag_words, nltk.MLEProbDist)
        emission_prob_df = pd.DataFrame(index=unique_tags_ls, columns=unique_words_ls, data=np.zeros((len_unique_tags, len_unique_words)))
        
        rowx,coly=emission_prob_df.shape
        for i in tqdm(range(int(rowx))):
            for j in range(int(coly)):
                emission_prob_df.at[unique_tags_ls[i],unique_words_ls[j] ]= cpd_tag_words[unique_tags_ls[i]].prob(unique_words_ls[j]) 
        
        return emission_prob_df

    def _eval_transition_probabilities(self):
        self.logger.info("Calculating Transition Probabilities...")
        cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(self.tags_sequence_ls))
        unique_tags_ls = list(self.unique_tags)
        len_unique_tags = len(self.unique_tags)
        
        transition_prob_df = pd.DataFrame(index=unique_tags_ls, columns=unique_tags_ls, 
                            data=np.zeros((len_unique_tags, len_unique_tags)))
        
        rowi,colj=transition_prob_df.shape
        for i in tqdm(range(rowi)):
            for j in range(colj):
                transition_prob_df.at[unique_tags_ls[i],unique_tags_ls[j] ]=(cfd_tags[unique_tags_ls[i]][unique_tags_ls[j]])/self.tags_sequence_ls.count(unique_tags_ls[i])
        
        return transition_prob_df
    
    def _is_training_file_present(self):
        return True if self.training_file else False

    def _convert_tuples(self):
        with open(self.training_file, "r") as tfile:        
            for line in tfile:
                line = line.strip(" \n")
                sentence_ls = []
                for word_tag in line.split(" "):
                    word,tag = word_tag.rsplit("/", 1)
                    sentence_ls.append((word, tag))
                yield sentence_ls
    
    def load_data(self):
        self.logger.info("Loading Train Data....")
        if self._is_training_file_present():
            tagged_sentences = self._convert_tuples()      
        else: 
            self._download_brown_corpus()
            tagged_sentences = corpus.tagged_sents(tagset='universal')
    
        for sentence in tagged_sentences: # get tagged sentences
            self.tag_word_tupl_ls.append( ("START", "START") )
            self.tags_sequence_ls.append("START")
            for (word, tag) in sentence:
                self.tags_sequence_ls.append(tag)
                self.tag_word_tupl_ls.append( (tag, word) ) 
                self.unique_words.add(word)
                self.unique_tags.add(tag)
            self.tag_word_tupl_ls.append(("END", "END"))
            self.tags_sequence_ls.append("END")

    def learn_training_probabilities(self):
        try:
            transition_prob_df = self._eval_transition_probabilities()
            emission_prob_df = self._eval_emission_probabilities()

            save_path = self.model_dump_path
            hdf = pd.HDFStore(save_path)
            hdf.put("transition_prob_df", transition_prob_df, data_columns=True)
            hdf.put("emission_prob_df", emission_prob_df, data_columns=True)
            self.logger.info("Saving model to {}".format(save_path))
            hdf.close()
        except Exception as ex:
            raise Exception("Error while Learning Probabilities!")


def get_args():
    parser = ArgumentParser(description='Train model for Parts of Speech Tagging.')
    parser.add_argument('-i', '--input_file', help='Input Train file. (Default: NLTK Brown Corpus)')
    parser.add_argument('-m', '--model_output_file', default="./model.h5", help='Model file Path to save model') 
    parser.add_argument("-v", "--verbose", help="increase output verbosity (Set Log Level: INFO)",
                    action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    training_filepath = args.input_file
    model_output_file = args.model_output_file
    
    hmm_learn = HMMLearn(training_filepath, model_output_file)
    hmm_learn.load_data()
    hmm_learn.learn_training_probabilities()   



