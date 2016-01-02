import unittest
from edmonds import edmonds

class EdmondsTests(unittest.TestCase):
           
    def test_basic_input(self):        
        G = {
            0: {1:10, 3:5},
            1: {2:1, 3:2},
            2: {4:4},
            3: {1:3, 2:9, 4:2},
            4: {0:7, 2:6}
        }
        root = 0
        self.assertDictEqual(edmonds.mst(root, G), {0: {3: 2}, 1: {2: 1}, 3: {1: 3, 4: 2}})

if __name__ == "__main__":
    unittest.main()