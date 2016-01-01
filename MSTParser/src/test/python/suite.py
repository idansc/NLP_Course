import unittest

from dataparser_tests import DataParserTests
from features_manager_tests import FeaturesManagerTests

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(DataParserTests))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(FeaturesManagerTests))
    return suite

if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())