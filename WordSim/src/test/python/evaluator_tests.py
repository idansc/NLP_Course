import unittest
from dataparser import Parser
from evaluator import Evaluator

class EvaluatorTests(unittest.TestCase):

    def test_matrices(self):
        parser = Parser('../resources/test/SimLex-999.csv', '../resources/test/wordsim353.csv', '../resources/test/test_corpus.txt')
        evaluator = Evaluator(parser)
        evaluator.evaluate()
        
        
if __name__ == "__main__":
    unittest.main()