# SparseNALG-0.1

Digital appendix for 'An Algebraic Generalization for Graph and Tensor-Based Neural Networks' in preparation for CIBCB 2017.  

This appendix includes an implementation of algebraic operations based on relational sums using linear algebraic primitives. Operations based on relational sums are used to build a useful extension to linear algebra called SparseNALG that enables the construction of arbitrary connectivity matrices for describing neural networks.

SparseNALG and related, future implementations are meant to provide a formal framework for neural network construction based on existing mathematical objects and methods, such as relation and linear algebras. Unlike other frameworks, such as the Connection Set Algebra, SparseNALG is based entirely on existing algebraic methods that are flexible, extensible, and formal. This means that new operations can be designed and reasoned about abstractly using established, familiar mathematics. 

Implementations of algebraic operations, starting with the relational sums, are found in [SparseNALG.py](SparseNALG.py). Examples of their use are found in [TotalNetExample.py](TotalNetExample.py). Also included are two Python scripts for converting connectivity matrices to [HyperNEAT Genomes](ToHyperNEATGenome.py) and Substrates(ToHyperNEATSubstrate.py).

## Dependencies
- Python 2  
- Numpy  
- Scipy  
- Matplotlib  
