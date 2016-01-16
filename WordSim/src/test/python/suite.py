import unittest

from dataparser_tests import DataParserTests

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(DataParserTests))
    return suite

if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())