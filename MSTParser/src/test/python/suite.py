import unittest

from dataparser_tests import DataParserTests
from features_manager_tests import FeaturesManagerTests
from edmonds_tests import EdmondsTests

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(DataParserTests))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(FeaturesManagerTests))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(EdmondsTests))
    return suite

if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())