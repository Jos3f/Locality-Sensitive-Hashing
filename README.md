# Locality-Sensitive-Hashing
Implementation of Locality Sensitive Hashing (LSH) for similarity comparison in large data sets.

This code was implemented and by [@jos3f](https://github.com/hkindbom/ID2222-Data-Mining) and  [@hkindbom](https://github.com/hkindbom). 

The implementation furthermore contains document comparison using Jaccard similarity of the shingle sets, and signature similarity comparison which is an approximation of the former measure of similarity. The Evaluation class contains a comparison of the three stated approaches for this task with respect to accuracy and time complexity. 

### How to run

Tested with Python 3.6 and 3.8. 

Requirements: See requirements.txt in zip file. Can be installed with pip: `pip install -r requirements.txt`

Run: `python main.py`. Use the flag -h for help: `python main.py -h`.  

Two sample data sets are located in the Data directory, one of which contains pairs of very similar documents, which are more or less identified using the different methods.

### Evaluation

### 