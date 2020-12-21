"""
A class CompareSets that computes the Jaccard similarity
of two sets of integers – two sets of hashed shingles.
"""

class CompareSets:

    def get_similarity(self, set_A, set_B):
        return self.jaccard_similarity(set_A, set_B)

    def jaccard_similarity(self, set_A, set_B):
        return len(set_A.intersection(set_B)) / len(set_A.union(set_B))


if __name__ == "__main__":
    # Example usage of this class
    set_A = {5, 4, 3, 1}
    set_B = {3, 0, 3, 1}
    compare_sets = CompareSets()
    print(compare_sets.get_similarity(set_A, set_B))