"""
To test and evaluate scalability (the execution time versus the size of
input dataset) of your implementation, write a program that uses your
classes to find similar documents in a corpus of 5-10 documents.
Choose a similarity threshold s (e.g., 0,8) that states that two documents
are similar
if the Jaccard similarity of their shingle sets is at least s.
"""


from DataLoader import DataLoader
from Shingling import Shingling
from CompareSets import CompareSets
from MinHashing import MinHashing
from CompareSignatures import CompareSignatures
from LSH import LSH

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from tqdm import tqdm
import time

class Evaluation:

    def __init__(self, small_file, large_file, nr_docs, char_dim, k, threshold, signature_len):
        """
        :param small_file: file path (string) to dataset with a few documents used to test algos on similarity
        :param large_file: file path (string) to large dataset of documents used to test scalability
        :param nr_docs: number of documents to read from the large dataset
        :param char_dim: int > 0, maximum length of documents
        :param k: int > 0, k in k-shingles (number of characters in each shingle)
        :param threshold: similarity threshold, float on interval [0, 1]
        :param signature_len: int > 0, dimension of signatures
        """
        self.large_docs = None
        self.small_docs = None
        self.nr_docs = nr_docs
        self.char_dim = char_dim
        self.k = k
        self.threshold = threshold
        self.signature_len = signature_len

        self.read_files(small_file, large_file)

    def read_files(self, small_file, large_file):
        small_data_loader = DataLoader(small_file)
        self.small_docs = small_data_loader.get_documents(10, randomize=False)

        large_data_loader = DataLoader(large_file)
        self.large_docs = large_data_loader.get_documents(self.nr_docs, self.char_dim)
        while len(self.large_docs) < self.nr_docs:
            self.large_docs.extend(self.large_docs)

    def find_similar_docs(self):
        """
        Computes similarity of all combinations of documents in small dataset and plots as bar chart for three methods.
        :return:
        """
        print("Finding similar documents: ")
        small_shingling = Shingling(self.small_docs, self.k)
        small_shingling.docs_to_hashed_shingles()
        shingle_sets = small_shingling.hashed_shingles

        min_hashing_object = MinHashing()
        signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets, signature_len=self.signature_len)
        y_sig = self.test_sig_similarity(signature_matrix)

        x, y_jac = self.test_jaccard_similarity(shingle_sets)
        similar_docs_LSH = self.find_similar_pairs_LSH(signature_matrix)[0]

        mean = (np.array(y_jac) - np.array(y_sig)).mean()
        var = (np.array(y_jac) - np.array(y_sig)).var()
        print(
            "\nMean and variance of difference between jaccard and signature similarity:\nMean: {}, variance: {}".format(
                mean, var))
        mean_abs_residual = (np.abs(np.array(y_jac) - np.array(y_sig))).mean()
        print(
            "\nMean absolute residual between jaccard and signature similarity:\nMean: {}".format(
                mean_abs_residual))
        self.bar_chart(x, y_jac, y_sig, similar_docs_LSH)

    def eval_scalability(self, nr_docs_list):
        """
        Takes a list of integers, nr documents to test run time with and then plots for all three algos
        :param nr_docs_list: List of number of documents to include in each step
        :return:
        """
        print("Evaluating scalability:")
        LSH_times = []
        min_hash_times = []
        pure_shingling_times = []
        for nr_docs in nr_docs_list:
            print("Starting for nr_docs: " + str(nr_docs))
            signature_matrix, sig_mat_time = self.get_signature_mat(nr_docs)
            LSH_times.append(self.find_similar_pairs_LSH(signature_matrix)[1] + sig_mat_time)
            min_hash_times.append(self.find_similar_pairs_min_hash(signature_matrix) + sig_mat_time)
            pure_shingling_times.append(self.find_similar_pairs_pure(nr_docs))

        self.plot_scalability(nr_docs_list, LSH_times, min_hash_times, pure_shingling_times)

    def compare_signature_lengths(self, sign_len_list):
        """
        Evaluate the precision of the approximated similarity with respect to signature length
        :param sign_len_list:  List of number of documents to include in each step
        :return:
        """
        print("Evaluating approximation precision:")
        signature_len_temp = self.signature_len
        mean_abs_residual_list = [0] * len(sign_len_list)

        small_shingling = Shingling(self.small_docs, self.k)
        small_shingling.docs_to_hashed_shingles()
        shingle_sets = small_shingling.hashed_shingles
        min_hashing_object = MinHashing()

        for i in tqdm(range(len(sign_len_list))):
            # print("Starting for signature length: " + str(i))

            signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets,
                                                                            signature_len=sign_len_list[i])
            y_sig = self.test_sig_similarity(signature_matrix)
            y_jac = self.test_jaccard_similarity(shingle_sets)[1]

            mean_abs_residual = (np.abs(np.array(y_jac) - np.array(y_sig))).mean()
            mean_abs_residual_list[i] = mean_abs_residual

        self.signature_len = signature_len_temp
        self.plot_mean(sign_len_list, mean_abs_residual_list)

    def plot_scalability(self, nr_docs_list, LSH_times, min_hash_times, pure_shingling_times):
        """
        Helper function for plotting.
        :param nr_docs_list:
        :param LSH_times:
        :param min_hash_times:
        :param pure_shingling_times:
        :return:
        """
        plt.figure()
        plt.plot(nr_docs_list, LSH_times, label='LSH')
        plt.plot(nr_docs_list, min_hash_times, label='MinHash sim')
        plt.plot(nr_docs_list, pure_shingling_times, label='Pure Shingling (Jaccard)')

        plt.title("Computation times for different methods", fontsize=18)
        plt.xlabel("nr docs", fontsize=14)
        plt.ylabel("Run time (seconds)", fontsize=14)
        plt.legend()

        plt.show()

    def get_signature_mat(self, nr_docs):
        """
        Computes signature matrix of nr_docs from large dataset and returns it together with computation time.
        :param nr_docs: Nr of documents to include from the large document set.
        :return:
        """
        start_time = time.time()

        large_shingling = Shingling(self.large_docs[:nr_docs], self.k)
        large_shingling.docs_to_hashed_shingles()
        shingle_sets = large_shingling.hashed_shingles

        min_hashing_object = MinHashing()
        signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets, signature_len=self.signature_len)

        return signature_matrix, time.time() - start_time

    def find_similar_pairs_LSH(self, signature_matrix):
        """
        Helper function for finding candidate pairs found with LSH and also returns computation time.
        :param signature_matrix: Signature matrix
        :return: Candidate pairs and comp. time
        """
        lsh = LSH(signature_matrix)
        lsh.find_b_and_r(self.threshold, band_method=1, confidence=0.95)

        start_time = time.time()
        candidates_idx = lsh.get_candidate_pairs()
        return candidates_idx, time.time() - start_time

    def find_similar_pairs_min_hash(self, signature_matrix):
        """
        Returning candidate pairs found with Min Hashing and also returns computation time
        :param signature_matrix: Signature matrix
        :return: comp. time
        """
        start_time = time.time()

        indices = list(range(signature_matrix.shape[1]))
        similar_pairs = []
        for pair_idx in combinations(indices, 2):
            similarity = CompareSignatures.similarity(signature_matrix[:, (pair_idx[0], pair_idx[1])])
            if similarity >= self.threshold:
                similar_pairs.append(pair_idx)

        return time.time() - start_time

    def find_similar_pairs_pure(self, nr_docs):
        """
        Using hashed shingles and exhaustive comparison with jaccard similarity
        :param nr_docs: nr of documents to compare from the large document set
        :return: comp. time
        """
        start_time = time.time()

        large_shingling = Shingling(self.large_docs[:nr_docs], self.k)
        large_shingling.docs_to_hashed_shingles()
        shingle_sets = large_shingling.hashed_shingles

        compare_sets = CompareSets()
        similar_pairs = []

        for set_pair in combinations(shingle_sets, 2):
            similarity = compare_sets.get_similarity(set_pair[0], set_pair[1])
            if similarity >= self.threshold:
                similar_pairs.append(set_pair)
        return time.time() - start_time

    def test_sig_similarity(self, signature_matrix):
        """
        Helper function. Get signature similarity between all possible pairs.
        :param signature_matrix: signature matrix
        :return: list of similarities.
        """
        indices = list(range(signature_matrix.shape[1]))
        y = []
        for pair_idx in combinations(indices, 2):
            similarity = CompareSignatures.similarity(signature_matrix[:, (pair_idx[0], pair_idx[1])])
            y.append(similarity)
        return y

    def test_jaccard_similarity(self, shingle_sets):
        """
        Helper function. Get jaccard similarity between all possible pairs.
        :param shingle_sets: List of shingle sets
        :return: x index and list of similarities.
        """
        compare_sets = CompareSets()
        x = list(range(len(list(combinations(shingle_sets, 2)))))
        y = []

        for set_pair in combinations(shingle_sets, 2):
            y.append(compare_sets.get_similarity(set_pair[0], set_pair[1]))
        return x, y

    def bar_chart(self, x, y_jac, y_sig, similar_docs_LSH):
        """
        Helper function for plotting.
        :param x:
        :param y_jac:
        :param y_sig:
        :param similar_docs_LSH:
        :return:
        """
        labels = [pair for pair in combinations(range(len(self.small_docs)), 2)]

        plt.bar(x, y_jac, alpha = 0.7, label='Pure shingles')
        plt.bar(x, y_sig, alpha = 0.7, label='MinHash dim ' + str(self.signature_len))
        plt.scatter([labels.index(pair) for pair in similar_docs_LSH], [self.threshold for _ in similar_docs_LSH],
                    marker="*",  edgecolor='black', linewidth=0.5, s=128, color="lime", label="LSH candidate",
                    zorder=3)

        plt.title("Similarity of document pair combinations", fontsize=20)
        plt.xlabel("pair", fontsize=14)
        plt.xticks(x, labels, rotation=75)
        plt.ylabel("Similarity", fontsize=14)
        plt.legend()

        plt.show()

    @staticmethod
    def plot_mean(sign_len_list, mean):
        """
        Helper function for plotting signature length vs mean.
        :param sign_len_list:
        :param mean:
        :return:
        """
        plt.figure()
        plt.plot(sign_len_list, mean, label='Mean of abs residuals')

        plt.title("The mean of the magnitude of the residual errors, as signature length increases.",
                  fontsize=12)
        plt.xlabel("Signature length", fontsize=12)
        plt.xscale('log', base=3)
        plt.ylabel("Error", fontsize=12)

        plt.show()
        return


if __name__ == "__main__":
    """
    Data sets used:
        small_file: The path to the file with 5-10 documents for evaluating similarities.
        large_file: The path to the file with many documents (1000+) for evaluating scalability.
    """
    small_file = 'Data/bbc-text-small.csv'
    large_file = 'Data/bbc-text.csv'

    """
    Parameters:
        char_dim: Length of the documents
        k: shingle length
        threshold: Threshold used for the LSH algorithm
        signature_length: Desired default of the signatures
        nr_docs_list: A list of number of documents used when evaluating the time complexity.
        sign_len_list: A list of signature lengths that are used to evaluate the precision of the approximated similarity. 
    """
    char_dim = 500
    k = 5
    threshold = 0.6
    signature_len = 100
    nr_docs_list = [5, 10, 20, 50, 100, 500, 1000, 2000, 4000]
    sign_len_list = [int(3**(x*0.1)) for x in range(40, 100, 1)]  # List of signatures to compare the approximation on


    """
    Run evaluations:
        1. Find and plot similarities between documents. 
        2. Evaluate the scalability with respect to number of documents.
        3. Compare the precision of the approximated similarity with respect to signature length 
    """
    eval = Evaluation(small_file, large_file, max(nr_docs_list), char_dim, k, threshold, signature_len)

    print("-----------------------------------------")
    eval.find_similar_docs()
    print("-----------------------------------------")
    eval.eval_scalability(nr_docs_list)
    print("-----------------------------------------")
    eval.compare_signature_lengths(sign_len_list)
    print("-----------------------------------------")
    print("Evaluation Done.")






