import unittest

from dataparser_tests import DataParserTests
from features_generator_tests import FeaturesGeneratorTests

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(DataParserTests))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(FeaturesGeneratorTests))
    return suite

if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())