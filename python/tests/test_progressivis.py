from pynene import Index
import numpy as np
import unittest
import time
import progressivis
from progressivis.table.api import PTable
import pandas as pd

def random_table(n=100, d=10, dtype=np.float32):
    arr = np.array(np.random.rand(n, d), dtype=dtype)
    df = pd.DataFrame(arr, columns=['_{}'.format(i) for i in range(d)])
    return PTable(name=None, data=df), arr

def get_row(x, pt):
    return np.array(list(x.loc[pt,:].to_dict(ordered=True).values())).reshape((1,x.shape[1]))


class PseudoArray(object):
    def __init__(self, array):
        self._array = array

    @property
    def shape(self):
        return self._array.shape

    def __getitem__(self, key):
        return self._array[key]

    def __setitem__(self, key, v):
        self._array[key] = v

    def __len__(self):
        return len(self._array)

class Test_Panene(unittest.TestCase):
    def test_return_shape(self):
        x,_ = random_table()
        index = Index(x)
        index.add_to_index(x.index.to_array())
        # self.assertIs(x, index.array)
        self.assertTrue(index.is_using_pyarray)

        index.add_points(x.shape[0])

        for i in range(x.shape[0]):
            ids, dists = index.knn_search(i, 5)
            self.assertEqual(ids.shape, (1, 5))
            self.assertEqual(dists.shape, (1, 5))

    def test_return_shape_64(self):
        x, _ = random_table(dtype=np.float64)

        index = Index(x)
        # self.assertIs(x, index.array)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(x.shape[0])

        for i in range(x.shape[0]):
            ids, dists = index.knn_search(i, 5)
            self.assertEqual(ids.shape, (1, 5))
            self.assertEqual(dists.shape, (1, 5))


    def test_random(self):
        x, _ = random_table()
        index = Index(x)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(x.shape[0]) # we must add points before querying the index

        pt = np.random.randint(x.shape[0])
        #pts = x[[pt]]
        pts = get_row(x, pt)
        idx, dists = index.knn_search_points(pts, 1, cores=1)
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx[0], pt)

    def test_random_64(self):
        x, _ = random_table(dtype=np.float64)
        index = Index(x)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(x.shape[0]) # we must add points before querying the index

        pt = np.random.randint(x.shape[0])
        row = get_row(x, pt)
        pts = np.asarray(row, dtype=np.float32)

        idx, dists = index.knn_search_points(pts, 1, cores=1)
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx[0], pt)


    def test_openmp(self):
        N = 10000 # must be large enough

        x, x_vec = random_table(N)
        index = Index(x)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(x.shape[0]) # we must add points before querying the index

        for r in range(5): # make cache ready
            idx, dists = index.knn_search_points(x_vec, 10)

        start = time.time()
        ids1, dists1 = index.knn_search_points(x_vec, 10, cores=1)
        elapsed1 = time.time() - start

        start = time.time()
        ids2, dists2 = index.knn_search_points(x_vec, 10, cores=4)
        elapsed2 = time.time() - start

        print("single thread: {:.2f} ms".format(elapsed1 * 1000))
        print("4 threads: {:.2f} ms".format(elapsed2 * 1000))

    def test_openmp_64(self):
        N = 10000 # must be large enough

        x, x_vec = random_table(N, dtype=np.float64)

        index = Index(x)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(x.shape[0]) # we must add points before querying the index
        pts = np.asarray(x_vec, dtype=np.float32)

        for r in range(5): # make cache ready
            idx, dists = index.knn_search_points(pts, 10)

        start = time.time()
        ids1, dists1 = index.knn_search_points(pts, 10, cores=1)
        elapsed1 = time.time() - start

        start = time.time()
        ids2, dists2 = index.knn_search_points(pts, 10, cores=4)
        elapsed2 = time.time() - start

        print("single thread: {:.2f} ms".format(elapsed1 * 1000))
        print("4 threads: {:.2f} ms".format(elapsed2 * 1000))

    @unittest.skip
    def test_openmp_obj(self):
        N = 10000 # must be large enough

        x0, x_vec = random_table(N, dtype=np.float64)
        x = PseudoArray(x_vec)

        index = Index(x)
        self.assertFalse(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(x.shape[0]) # we must add points before querying the index

        pts = np.asarray(x_vec, dtype=np.float32)

        for r in range(5): # make cache ready
            idx, dists = index.knn_search_points(pts, 10)

        start = time.time()
        ids1, dists1 = index.knn_search_points(pts, 10, cores=1)
        elapsed1 = time.time() - start

        start = time.time()
        ids2, dists2 = index.knn_search_points(pts, 10, cores=4)
        elapsed2 = time.time() - start

        print("single thread: {:.2f} ms".format(elapsed1 * 1000))
        print("4 threads: {:.2f} ms".format(elapsed2 * 1000))


    def test_large_k(self):
        x, _ = random_table()
        _, q = random_table(1)
        k = x.shape[0] + 1 # make k larger than # of vectors in x

        index = Index(x)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(x.shape[0])

        with self.assertRaises(ValueError):
            index.knn_search(0, k)

        with self.assertRaises(ValueError):
            index.knn_search_points(q, k)

    def test_incremental_run1(self):
        x, _ = random_table()

        index = Index(x, w=(0.5, 0.5))
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        ops = 20

        for i in range(x.shape[0] // ops):
            ur = index.run(ops)

            self.assertEqual(index.size(), (i + 1) * ops)
            self.assertEqual(ur['addPointResult'], ops)

    def test_incremental_run2(self):
        n = 1000
        k = 20
        ops = 100
        test_n = 30

        x, _ = random_table(n)
        _, test_points = random_table(test_n)

        index = Index(x)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        for i in range(n // ops):
            ur = index.run(ops)

            ids1, dists1 = index.knn_search_points(test_points, k, checks = 100)
            ids2, dists2 = index.knn_search_points(test_points, k, checks = 1000)

            """
            The assertion below always holds since the latter search checks a larger number of nodes and the search process is deterministic
            """
            self.assertEqual(np.sum(dists1 >= dists2), test_n * k)

    def test_check_x_type(self):
        x, x_vec = random_table()
        index = Index(x)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(len(x))
        index.knn_search_points(x_vec, 10)

        with self.assertRaises(ValueError):
            x, x_vec = random_table(dtype=np.int32)
            index = Index(x)
            index.add_to_index(x.index.to_array())
            index.add_points(len(x))
            index.knn_search_points(x_vec, 10)

        with self.assertRaises(ValueError):
            x, x_vec = random_table(dtype=np.float64)
            index = Index(x)
            index.add_to_index(x.index.to_array())
            index.add_points(len(x))
            index.knn_search_points(x_vec, 10)
    @unittest.skip
    def test_updates_after_all_points_added(self):
        np.random.seed(10)
        n = 10000
        w = (0.5, 0.5)
        x, _ = random_table(n)
        ops = 1000

        index = Index(x, w=w)
        self.assertTrue(index.is_using_pyarray)
        index.add_to_index(x.index.to_array())
        index.add_points(n) # add all points

        for i in range(1000):
            index.knn_search_points(random_table(100)[1], 10) # accumulate losses

        for i in range(10):
            res = index.run(ops)

            self.assertEqual(res['addPointResult'], 0)
            self.assertEqual(res['updateIndexResult'], ops)

if __name__ == '__main__':
    unittest.main()
