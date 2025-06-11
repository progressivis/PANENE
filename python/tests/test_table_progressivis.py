from pynene import KNNTable
import numpy as np
import unittest
import progressivis
from progressivis.table.api import PTable
from progressivis.core.pintset import PIntSet
import pandas as pd


def random_table(n=100, d=10, dtype=np.float32):
    arr = np.array(np.random.rand(n, d), dtype=dtype)
    df = pd.DataFrame(arr, columns=['_{}'.format(i) for i in range(d)])
    return PTable(name=None, data=df), arr

def get_row(x, pt):
    return np.array(list(x.loc[pt,:].to_dict(ordered=True).values())).reshape((1,x.shape[1]))



class Test_KNNTable(unittest.TestCase):
    def test_table(self):
        n = 100
        d = 10
        k = 5
        array, _ = random_table(n, d)
        neighbors = np.zeros((n, 5), dtype=np.int64) # with np.in32, it is not optimized
        distances = np.zeros((n, 5), dtype=np.float32)
        table = KNNTable(array, k, neighbors, distances)
        
        self.assertTrue(table != None)
        updates = table.run_ids(PIntSet(range(10)).to_array())
        #print(updates)
        self.assertEqual(len(updates), 11) # 10+1empty set

    def test_incremental_run(self):
        n = 1000
        ops = 100
        k = 20
        neighbors = np.zeros((n, k), dtype=np.int64)
        distances = np.zeros((n, k), dtype=np.float32)
        x, _ = random_table(n)
        table = KNNTable(x, k, neighbors, distances)
        for i in range(n // ops):
            lower = i*ops
            upper = lower+ops
            ids = PIntSet(x.id_to_index(x.index.value[lower:upper])).to_array()
            ur = table.run_ids(ids)
            for nn in range(ur['numPointsInserted']):
                for kk in range(k - 1):
                    self.assertTrue(distances[nn][kk] <= distances[nn][kk+1])
                    
                for kk in range(k):
                    idx = neighbors[nn][kk]
                    self.assertAlmostEqual(distances[nn][kk], np.sum((get_row(x, nn) - get_row(x, idx)) ** 2) ** 0.5, places=3)

    def test_updates_after_all_points_added(self):
        np.random.seed(10)
        n = 10000
        w = (0.5, 0.5)
        x, _ = random_table(n)
        ops = 1000
        k = 10

        neighbors = np.zeros((n, k), dtype=np.int64)
        distances = np.zeros((n, k), dtype=np.float32)

        table = KNNTable(x, k, neighbors, distances)

        for i in range(200):
            lower = i*ops
            upper = lower+ops
            ids = PIntSet(x.id_to_index(x.index[lower:upper])).to_array()            
            ur = table.run_ids(ids)

            if ur['numPointsInserted'] >= n:
                break

        for i in range(10):
            lower = i*ops
            upper = lower+ops
            ids = PIntSet(x.id_to_index(x.index.value[lower:upper])).to_array()            
            ur = table.run_ids(ids)

            self.assertTrue(ur['addPointOps'] + ur['updateIndexOps'] <= w[0] * ops)

if __name__ == '__main__':
    unittest.main()
