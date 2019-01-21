import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from utils import *

class AnswerRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
#         self.question_matrix = unpickle_file(paths['QUESTION_MATRIX'])
    
    def get_best_answer(self, question, question_matrix):
        """ 
        Returns id of the most similar thread for the question.
        The search is performed across the threads with a given tag.
        """
        question = text_prepare(question)
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_answer_id = pairwise_distances_argmin(question_vec.reshape(1, -1), question_matrix)[0]
        question_embedding = question_matrix[best_answer_id]
        
        return best_answer_id, question_embedding
    
    def get_final_answer(self, question, path):
        id_list, embedding_list = [], []
        matrix_size = 100000
        for dir_path, dirs, files in os.walk(path):
            for i, filename in enumerate(sorted(files, key=lambda x: int(re.findall('\d+', x)[0]))):
                file_path = os.path.join(dir_path, filename)
                question_matrix = unpickle_file(file_path)
                best_id, best_embedding = self.get_best_answer(question, question_matrix)
                del question_matrix
                best_id = best_id + i*matrix_size
                id_list.append(best_id)
                embedding_list.append(best_embedding)
    
        final_id, _ = self.get_best_answer(question, embedding_list)
    
        return id_list[final_id]
                        
                        
class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")
        
        self.answer_ranker = AnswerRanker(paths)
        self.answer_pairs = self.get_answer_pairs(paths['ANSWER_PAIRS'])

    def generate_answer(self, question):
        answer_id = self.answer_ranker.get_final_answer(question, RESOURCE_PATH['QUESTION_MATRIX'])
        
        return self.answer_pairs[answer_id]
        
    def get_answer_pairs(self, answer_path):
        answer_pairs = []
        with open(answer_path, 'r') as f:
            for line in f:
                line = line.strip()
                answer_pairs.append(line)
        
        return answer_pairs
        