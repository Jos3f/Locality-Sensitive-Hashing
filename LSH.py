"""
A class LSH that implements the LSH technique: given a collection of minhash signatures (integer vectors) and a similarity threshold t, the LSH class (using banding and hashing) finds all candidate pairs
of signatures that agree on at least fraction t of their components.
"""

class LSH:

    def __init__(self, signature_matrix):
        """
        Initialize with a signature matrix.
        :param signature_matrix: n*c numpy matrix where n is the signature length and c is num documents.
        """
        self.signature_matrix = signature_matrix
        self.signature_matrix.flags.writeable = False
        self.b = 0
        self.r = 0

    def find_b_and_r(self, threshold, band_method=1, confidence=0.95):
        """
        Finds and sets band and row numbers for this object. There are three methods in this implementation:
            0: Select bands such that threshold = (1/bands)^(1/rows)
            1: Select bands such confidence = 1 - (1 - threshold^rows)^bands
            2: Use the Pigeonhole principle to ensure all signatures that coincide on the given threshold. This method
            may result in many false negatives.
        All these methods ensures that bands*rows=signature length

        :param threshold: The similarity threshold for candidate pairs
        :param band_method: Method for selecting bands.
        :param confidence: Portion of how many with the given threshold similarity we should include. This is only
        needed if band_method 1 is used.
        :return:
        """
        if band_method == 0:
            b, r = self._get_best_approx_band(threshold)
        elif band_method == 1:
            b, r = self._get_best_confident_band(threshold, confidence)
        else:
            b, r = self._get_best_confident_band(threshold)
        self.b = b
        self.r = r


    def get_candidate_pairs(self):
        """
        Get a list of candidate pairs. Each pair is represented as a tuple of indices.
        :return: list of candidate pairs
        """
        candidate_pairs = set() # Set of candidate pairs
        for b_idx in range(self.b):
            # Iterate over every band
            boxes = dict() # Create an empty dictionary where we collect boxes and keep track of collisions
            for doc in range(self.signature_matrix.shape[1]):
                # Iterate over every document for the current band and calculate which box it belongs to by hashing it.
                box = hash(tuple(self.signature_matrix[b_idx*self.r:(b_idx+1)*self.r, doc]))
                if box in boxes:
                    for doc_in_box in boxes[box]:
                        # Create candidate pair if two documents are placed in the same box.
                        candidate_pairs.add((doc_in_box, doc))
                    boxes[box].add(doc)
                else:
                    # Create box in dictionary if the document is first in this box.
                    boxes[box] = {doc}
        return candidate_pairs

    def _get_best_confident_band(self, threshold, confidence=0.95):
        """
        Chooses number of bands such that the recall is equal to, or greater than a set confidence
        :param threshold: The similarity threshold.
        :param confidence: The desired recall.
        :return: bands, rows
        """
        # Calculation: confidence = 1 - (1 - t^r)^b
        signature_length = self.signature_matrix.shape[0]
        b = signature_length
        r = signature_length / b
        b_hat = b
        r_hat = r
        confidence_hat = 1 - (1 - threshold**r)**b
        # exhaustive search, test all pairs band and rows, ensuring bands*rows=signature length
        while b > 0:
            if signature_length % b != 0:
                b -= 1
                continue
            r = signature_length / b
            confidence_temp = 1 - (1 - threshold**r)**b
            if abs(confidence_hat - confidence) > abs(confidence_temp - confidence) and confidence_temp > confidence:
                b_hat = b
                r_hat = r
                confidence_hat = confidence_temp
            b -= 1

        return b_hat, int(r_hat)

    def _get_best_band_ensuring_t(self, threshold):
        """
        Based on the Pigeonhole principle.
        Ensures no false negatives, but is very inclined to including false positives.
        :param threshold:
        :return: bands, rows
        """
        # Calculate smallest number of pigeonholes to ensure that each can not contain a different element
        signature_len = self.signature_matrix.shape[0]
        eps = 1e-9 # Needed for floating point errors
        pigeonholes = int(signature_len * (1 - threshold) + 1 + eps)
        # Get the smallest number of pigeonholes that is equal to or larger than required,
        # while still being a factor of the signature length
        while pigeonholes <= signature_len:
            if signature_len % pigeonholes == 0:
                break
            pigeonholes += 1

        return pigeonholes, int(signature_len / pigeonholes)

    def _get_best_approx_band(self, threshold):
        """
        Approximates number of bands according to: threshold = (1/b)^(1/r)
        :param threshold: Threshold
        :return: bands, rows
        """
        # threshold = (1/b)^(1/r)
        # ensure  n = r * b
        # optimize, find r * b such that (t - (1/b)^(1/r)) is minimized
        signature_length = self.signature_matrix.shape[0]
        b = signature_length
        r = signature_length / b
        b_hat = b
        r_hat = r
        threshold_hat = (1 / b) ** (1 / r)
        # exhaustive search, test all pairs band and rows, ensuring bands*rows=signature length
        while b > 0:
            if signature_length % b != 0:
                b -= 1
                continue
            r = signature_length / b
            threshold_temp = (1 / b) ** (1 / r)
            if abs(threshold_hat - threshold) > abs(threshold_temp - threshold):
                b_hat = b
                r_hat = r
                threshold_hat = threshold_temp
            b -= 1
        return b_hat, int(r_hat)


def main():
    # Example usage of this class
    from MinHashing import MinHashing
    min_hashing_object = MinHashing()
    example_shingles = [{0, 1, 5, 6}, {2, 3, 4}, {0, 5, 6},  {1, 2, 3, 4}]
    signature_matrix = min_hashing_object.get_signature_matrix_hash(example_shingles, signature_len=100)

    # lsh = LSH(np.zeros((100, 4)), 0.8)
    lsh = LSH(signature_matrix)
    print(lsh._get_best_approx_band(0.8))
    print(lsh._get_best_confident_band(0.8, confidence=0.95))
    print(lsh._get_best_band_ensuring_t(0.8))
    lsh.find_b_and_r(0.8)
    candidates = lsh.get_candidate_pairs()


if __name__ == '__main__':
    main()
