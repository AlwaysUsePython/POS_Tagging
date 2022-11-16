from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import logging

class HMM_Client:
    logger = logging.getLogger(__name__)
    def __init__(self, test_in_sentences:list, output_file:str, saved_model_path:str, ):
        self.saved_model_path = saved_model_path
        self.test_in_sentences = test_in_sentences or []
        self.output_file = output_file
        self.emission_prob_df = None
        self.transition_prob_df = None
        self.results = []
        self.pos_tags = []
        
    def check_model_exists(self):
        if not os.path.exists(self.saved_model_path):
            raise FileNotFoundError("Model file not found! Please pass file path using '-m' option")
        return True

    def _formate_output(self, output_pos_tags, original_sentence):
        formated_ls = []
        sentence_ls = original_sentence.split(" ")
        for i in range(len(sentence_ls)):
            formated_ls.append("{0}/{1}".format(sentence_ls[i],output_pos_tags[i]))
        
        return ' '.join(formated_ls)

    def _hmm_viterbi_algo(self, words:list, observations_len:int, states:list, states_len:int):
        viterbi = np.array([[0.0]*observations_len]*states_len) #states_len+2,observations_len
        backpointer = np.array([[-1]*observations_len]*states_len)
        
        for s in range(1,states_len-1):
            ans = self.transition_prob_df.loc['START'][states[s]]*self.emission_prob_df.loc[states[s]][words[0]]
            viterbi[s][0] = ans
            backpointer[s][0] = 0
        
        for step in range(1, observations_len):
            
            for state_no in range(1, states_len-1):
                max_prob = -1
                prev_state = -1
                for pstate_no in range(1, states_len-1):
                    current_prob = viterbi[pstate_no][step-1]*self.transition_prob_df.loc[states[pstate_no]][states[state_no]]*self.emission_prob_df.loc[states[state_no]][words[step]]
                    if current_prob>max_prob:
                        max_prob = current_prob
                        prev_state = pstate_no
                    
                viterbi[state_no][step] = max_prob
                backpointer[state_no][step] = prev_state 

        #Backtrace backpointer array
        last_word_state_index = np.argmax(viterbi, axis=0)[observations_len-1]
        res_pos_tag_ls = [self.pos_tags[last_word_state_index]]
        current_state_index = last_word_state_index
        current_observations_index = observations_len -1

        while backpointer[current_state_index][current_observations_index]!=0:
            res_pos_tag_ls.insert(0, self.pos_tags[backpointer[current_state_index][current_observations_index]])
            current_state_index = backpointer[current_state_index][current_observations_index]
            current_observations_index-=1

        return res_pos_tag_ls

    def apply_hmm_viterbi(self):
        self.logger.info("Running HMM on Test Sentences")
        self._update_pos_tags()
        with open(self.output_file, "w") as op_f: 
            for t_sentence in self.test_in_sentences :
                try:
                    words = t_sentence.split(" ")

                    res_pos_tags = self._hmm_viterbi_algo(words, len(words), self.pos_tags, len(self.pos_tags))
                    res_str = self._formate_output(res_pos_tags, t_sentence)
                    self.results.append(res_str)
                    op_f.write(res_str)
                    op_f.write("\n")
                except  KeyError as kex:
                    self.logger.info("Word not found in vocab. Ignoring sentence '{}' ".format(t_sentence))
                except Exception as ex:
                    self.logger.error(ex)
    
    def read_model(self):
        try:
            hdf = pd.HDFStore(self.saved_model_path, mode='r')

            self.emission_prob_df = hdf.get('/emission_prob_df')
            self.transition_prob_df = hdf.get('/transition_prob_df')
            hdf.close()
        except Exception as ex:
            self.logger.error(ex)
            raise Exception("Error while reading model file.")
    
    def _update_pos_tags(self):
        self.pos_tags = self.transition_prob_df.index.tolist()
        
        self.pos_tags.remove('START')
        self.pos_tags.remove('END')

        self.pos_tags.insert(0, 'START')
        self.pos_tags.append('END')
    
    def print_results(self):
        print("")
        for ele in self.results:
            print(ele)
    
        return "Hello"

def get_args():
    parser = ArgumentParser()
    parser.add_argument('sentences', help='Sentence OR Test file path contaning list of input sentences (Each at newline)')
    parser.add_argument('-o', '--output', default="./test.txt", help='Test Output file')
    parser.add_argument('-f', '--from_file', action='store_true',
            help='Read Input Sentences to train from a file') 
    parser.add_argument("-m", "--model", default="./model.h5", help="Model file path ")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    return parser.parse_args()

def main():
    args = get_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    sentences_ls = []
    if args.from_file:
        with open(args.sentences) as f:
            for sentence in f:
                sentence = sentence.strip(" \n")
                sentences_ls.append(sentence)
    else:
        sentences_ls.append(args.sentences)

    hmm = HMM_Client(sentences_ls, args.output, args.model)
    
    if hmm.check_model_exists():
        hmm.read_model()
        hmm.apply_hmm_viterbi()
        hmm.print_results()
        

if __name__ == '__main__':
    main()