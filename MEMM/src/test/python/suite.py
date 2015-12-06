import unittest

from dataparser_tests import DataParserTests
from features_manager_tests import FeaturesManagerTests
from optimizer_tests import OptimizerTests

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(DataParserTests))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(FeaturesManagerTests))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(OptimizerTests))
    return suite

if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())