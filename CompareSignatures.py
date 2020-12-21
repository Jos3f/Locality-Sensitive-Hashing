"""
A class CompareSignatures that estimates similarity of two integer vectors –
minhash signatures – as a fraction of components, in which they agree.
"""
import numpy as np


class CompareSignatures:

    def __init__(self):
        pass

    @ staticmethod
    def similarity(signatures):
        """
        Estimates similarity of two integer vectors as a fraction of components, in which they agree.
        :param signatures: A Nx2 matrix where N is the signature length and each column corresponds to a signature
        :return:
        """
        return np.count_nonzero(signatures[:, 0] == signatures[:, 1]) / signatures.shape[0]


def main():
    # Example use
    from MinHashing import MinHashing
    min_hashing_object = MinHashing()
    example_shingles = [{0, 1, 5, 6}, {2, 3, 4}, {0, 5, 6},  {1, 2, 3, 4}]
    signature_matrix = min_hashing_object.get_signature_matrix_hash(example_shingles, signature_len=10000)

    similarity = CompareSignatures.similarity(signature_matrix[:, (0, 2)])
    print(similarity)
    return


if __name__ == "__main__":
    main()