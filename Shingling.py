"""
A class Shingling that constructs k–shingles of a given length k (e.g., 10) from a given document,
computes a hash value for each unique shingle,
and represents the document in the form of a set of its hashed k-shingles.
"""

class Shingling:
    def __init__(self, documents, k=10, hash_power=32):
        """
        :param documents: list of documents (strings)
        :param k: int
        :param hash_power: int
        :return list of sets, each set being hashed k-shingles from a document
        """
        self.documents = documents
        self.hashed_shingles = []
        self.hash_power = hash_power
        self.k = k

    def docs_to_hashed_shingles(self):
        """
        Convert each document to a set of hash values (representing k-shingles), stored in a list
        :return:
        """
        for document in self.documents:
            shingle_set = self.doc_to_shingles(document)
            hashed_shingle_set = self.hash_shingles(shingle_set)
            self.hashed_shingles.append(hashed_shingle_set)

    def doc_to_shingles(self, document):
        """
        Create a set of unique k-shingles on character level
        :param document: a document as a string
        :return:
        """
        document = document.replace(' ', '_')
        shingles_zip = zip(*[document[i:] for i in range(self.k)])
        shingles_duplicates = [''.join(shingle) for shingle in shingles_zip]
        # set contains only unique shingles
        return set(shingles_duplicates)

    def hash_shingles(self, shingle_set):
        """
        Convert strings to integers with hashing on the range [0, 2^hash_power - 1)
        :param shingle_set: Shingle set where each shingle is represented as a hashable type.
        :return:
        """
        hashed_shingle_set = set()
        for shingle in shingle_set:
            hash_value = hash(shingle) % (2 ** self.hash_power - 1)
            hashed_shingle_set.add(hash_value)
        return hashed_shingle_set


if __name__ == "__main__":
    # Example usage of this class
    documents = ['name is', 'my name is Alice']
    k = 4

    shingling = Shingling(documents, k)
    shingling.docs_to_hashed_shingles()
    print(shingling.hashed_shingles)

