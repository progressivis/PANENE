from cpython cimport PyObject, Py_INCREF, array
from cython.operator import dereference

import warnings
import sys
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef char* version = '0.0.1'
cdef object __version__ = version

cdef inline check_array(arr):
    shape = arr.shape
    if len(shape)!=2: # add more tests
        raise TypeError('value is %s not an array', arr)
    if len(arr) != shape[0]:
        raise TypeError('Inconsistency in array len', arr)


cdef  vector[int32_t] add_to_index_impl(ids):
    cdef array.array a = <array.array?>ids
    cdef vector[int32_t] v =vector[int32_t]()
    v.reserve(len(ids))
    for i in ids:
        v.push_back(i)
    return v

cdef class Index:
    cdef SourceABC * c_src
    cdef IndexParams    c_indexParams
    cdef IndexABC    * c_index
    cdef table
    def __cinit__(self, array, w=(0.3, 0.7), float reconstruction_weight=0.25, trees = None):
        cdef PyDataSource_* src
        cdef ProgressivisSource_* src_pvs

        if not (hasattr(array, '__module__') and 'progressivis.table.' in array.__module__):
            check_array(array)
            src = new PyDataSource_(array)
            self.c_src = <SourceABC*>(new PyDataSource(src))
            self.c_index = <IndexABC*>(new PyIndexL2(new PyIndexL2_(src,
                                     self.c_indexParams, TreeWeight(w[0], w[1]),
                                     reconstruction_weight)))
        else:
            self.table = array
            array_ = self.table.get_panene_data()
            src_pvs = new ProgressivisSource_(array_)
            self.c_src = <SourceABC*>(new ProgressivisSource(src_pvs))
            self.c_index = <IndexABC*>(new PyIndexPvs(new PyIndexPvs_(src_pvs,
                                     self.c_indexParams, TreeWeight(w[0], w[1]),
                                     reconstruction_weight)))

        if trees is not None:
            self.c_indexParams.trees = trees

    def __dealloc__(self):
        del self.c_index
        del self.c_src

    @property
    def array(self):
        return self.c_src.get_array()

    @array.setter
    def array(self, value):
        check_array(value)
        self.c_src.set_array(value)

    @property
    def is_using_pyarray(self):
        return self.c_src.is_using_pyarray()

    def add_points(self, size_t end):
        self.refresh()
        self.c_index.addPoints(end)

    def size(self):
        return self.c_index.getSize()

    def refresh(self):
        if self.table is None:
            return
        array_ = self.table.get_panene_data()
        self.c_src.set_array(array_)

    def knn_search(self, int pid, size_t k, checks=None, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()

        if checks is not None:
            params.checks = checks
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if not self.is_using_pyarray:
            if cores is not None and cores != 1:
                warnings.warn('Ignoring cores for non-numpy arrays')
            cores = 1
        if cores is not None:
            params.cores = cores

        if self.c_index.getSize() < k:
            raise ValueError('k is larger than the number of points in the index. Make sure you called add_points()')

        cdef PyResultSet res = PyResultSet(k)
        self.refresh()
        with nogil:
            self.c_index.knnSearch(pid, res, k, params)

        ids = np.ndarray((1, res.k), dtype=np.int32) # was np.int
        dists = np.ndarray((1, res.k), dtype=np.float32)  # was np.float

        for i in range(k):
            nei = res[i]
            ids[0, i] = nei.id
            dists[0, i] = nei.dist

        return ids, dists

    def knn_search_points(self, np.ndarray[DTYPE_t, ndim=2] points, size_t k, checks=None, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()
        if checks is not None:
            params.checks = checks
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if not self.is_using_pyarray:
            if cores is not None and cores != 1:
                warnings.warn('Ignoring cores for non-numpy arrays')
                cores = 1
        if cores is not None:
            params.cores = cores

        if self.c_index.getSize() < k:
            raise ValueError('k is larger than the number of points in the index. Make sure you called add_points()')

        cdef size_t n = points.shape[0] # of query points
        cdef size_t d = points.shape[1] # dimension
        cdef Points cpoints = Points()

        cpoints.reserve(n)

        for j in range(n):
            cpoints.emplace_back(d)
            for i in range(d):
                cpoints[j][i] = points[j, i]

        cdef PyResultSets ress = PyResultSets(n)
        self.refresh()
        with nogil:
            self.c_index.knnSearchVec(cpoints, ress, k, params)
        ids = np.ndarray((n, k), dtype=np.int32)  # was np.int
        dists = np.ndarray((n, k), dtype=np.float32)
        cdef PyResultSet res

        for j in range(n):
            res = ress[j]
            for i in range(k):
                nei = res[i]
                ids[j][i] = nei.id
                dists[j][i] = nei.dist

        return ids, dists

    def add_to_index(self, ids):
        cdef vector[int32_t] v = add_to_index_impl(ids)
        self.c_src.add_to_index(v)

    def run(self, int ops):
        cdef UpdateResult2 ur
        self.refresh()
        with nogil:
            ur = self.c_index.run(ops)

        return {
            'numPointsInserted': ur.numPointsInserted,
            'addPointOps': ur.addPointOps,
            'updateIndexOps': ur.updateIndexOps,
            'addPointResult': ur.addPointResult,
            'updateIndexResult': ur.updateIndexResult,
            'addPointElapsed': ur.addPointElapsed,
            'updateIndexElapsed': ur.updateIndexElapsed
        }

    def run_ids(self, ids):
        self.add_to_index(ids)
        return self.run(len(ids))

cdef class KNNTable:
    cdef SourceABC * c_src
    cdef IndexParams    c_indexParams
    cdef SearchParams   c_searchParams
    cdef PyDataSink   * c_sink
    cdef KNNTableABC   * c_table
    cdef table

    def __cinit__(self, object array, int k, object neighbors, object distances,
                  treew=(0.3, 0.7), tablew=(0.5, 0.5),
                  trees=None,
                  checks=None, eps=None, sorted=None, cores=None
                  ):
        cdef PyDataSource_* src
        cdef ProgressivisSource_* src_pvs
        cdef progressivis_mode = False
        check_array(neighbors)
        check_array(distances)
        if neighbors.shape[1] != k or distances.shape[1] != k:
            raise ValueError('neighbors and distances should have axis=1 of %d'%k)

        if trees is not None:
            self.c_indexParams.trees = trees

        if checks is not None:
            self.c_searchParams.checks = checks
        if eps is not None:
            self.c_searchParams.eps = eps
        if sorted is not None:
            self.c_searchParams.sorted = sorted
        if not (hasattr(array, '__module__') and 'progressivis.table.' in array.__module__):
            check_array(array)
            src = new PyDataSource_(array)
            self.c_src = <SourceABC*>(new PyDataSource(src))
        else:
            progressivis_mode = True
            self.table = array
            array_ = self.table.get_panene_data()
            src_pvs = new ProgressivisSource_(array_)
            self.c_src = <SourceABC*>(
                new ProgressivisSource(src_pvs))
        self.c_sink = new PyDataSink(neighbors, distances)
        if not (self.is_using_pyarray and \
                self.is_using_neighbors_pyarray and \
                self.is_using_distances_pyarray):
            if cores is not None and cores != 1:
                warnings.warn('Ignoring cores for non-numpy arrays')
            cores = 1
        if cores is not None:
            self.c_searchParams.cores = cores
        if not progressivis_mode:
            self.c_table = <KNNTableABC*>(new PyKNNTable(new PyKNNTable_(src,
                                        self.c_sink,
                                        k,
                                        self.c_indexParams,
                                        self.c_searchParams,
                                        TreeWeight(treew[0], treew[1]),
                                        TableWeight(tablew[0], tablew[1])
                                        )))
        else:
            self.c_table = <KNNTableABC*>(new PyKNNTablePvs(new PyKNNTablePvs_(src_pvs,
                                        self.c_sink,
                                        k,
                                        self.c_indexParams,
                                        self.c_searchParams,
                                        TreeWeight(treew[0], treew[1]),
                                        TableWeight(tablew[0], tablew[1])
                                        )))

    @property
    def is_using_pyarray(self):
        return self.c_src.is_using_pyarray()

    @property
    def is_using_neighbors_pyarray(self):
        return self.c_sink.is_using_neighbors_pyarray()

    @property
    def is_using_distances_pyarray(self):
        return self.c_sink.is_using_distances_pyarray()

    def __dealloc(self):
        del self.c_table
        del self.c_src
        del self.c_sink

    def size(self):
        return self.c_table.getSize()
    @property
    def array(self):
        return self.c_src.get_array()

    def refresh(self):
        if self.table is None:
            return
        array_ = self.table.get_panene_data()
        self.c_src.set_array(array_)

    def add_to_index(self, ids):
        cdef vector[int32_t] v = add_to_index_impl(ids)
        self.c_src.add_to_index(v)

    def run(self, size_t ops):
        cdef UpdateResult ur
        self.refresh()
        with nogil:
            ur = self.c_table.run(ops)
        return {
            'addPointOps': ur.addPointOps,
            'updateIndexOps': ur.updateIndexOps,
            'updateTableOps': ur.updateTableOps,
            'addPointResult': ur.addPointResult,
            'updateIndexResult': ur.updateIndexResult,
            'updateTableResult': ur.updateTableResult,
            'numPointsInserted': ur.numPointsInserted,
            'addPointElapsed': ur.addPointElapsed,
            'updateIndexElapsed': ur.updateIndexElapsed,
            'updateTableElapsed': ur.updateTableElapsed,
	    'updatedIds': ur.updatedIds,
            }

    def run_ids(self, ids):
        self.add_to_index(ids)
        return self.run(len(ids))

# https://stackoverflow.com/questions/75626349/how-to-wrap-stdshared-ptr-and-stdvector-from-c-in-cython
# https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html

cdef class ProgressiVisTSNE:
    cdef PyDataSource_* c_src
    cdef ResponsiveTSNE * rtsne
    cdef Config * _config
    def __cinit__(self, table, column, skip_random=False, **kw):
        self.init_conf(**kw)
        arr = table[column].value
        mn = np.mean(arr, axis=0)
        norm = arr -mn
        max_ = np.max(np.fabs(norm))
        self.c_src = new PyDataSource_(norm/max_) #table[column].value)
        cdef vector[int32_t] ids = add_to_index_impl(table.index.to_array())
        self.c_src.add_to_index(ids)
        self.rtsne = new ResponsiveTSNE(self.c_src, skip_random, self._config)
        #self.rtsne.initialize(self.c_src, skip_random, self._config)
        #self.rtsne.run_ids(ids)

    def get_y(self):
        cdef vector[double] y = self.rtsne.Y
        cdef int ndims = self._config.output_dims
        np_vect = np.array(y, dtype=np.float64)
        return np_vect.reshape(-1, ndims)
    def get_error(self):
        cdef double err = self.rtsne.C
        return err

    def run_ids(self, ids):
        self.c_src.add_to_index(ids)
        return self.rtsne.run_ids(ids)
    def dump_y(self):
        self.rtsne.dump_Y()
    def init_conf(self, **kw):
        cdef Config * config = new Config()
        if "n" in kw:
            config.n = kw["n"]
        if "input_dims" in kw:
            config.input_dims = kw["input_dims"]
        if "output_dims" in kw:
            config.output_dims = kw["output_dims"]
        if "theta" in kw:
            config.theta = kw["theta"]
        if "perplexity" in kw:
            config.perplexity = kw["perplexity"]
        if "max_iter" in kw:
            config.max_iter = kw["max_iter"]
        if "use_ee" in kw:
            config.use_ee = kw["use_ee"]
        if "ee_factor" in kw:
            config.ee_factor = kw["ee_factor"]
        if "ee_iter" in kw:
            config.ee_iter = kw["ee_iter"]
        if "use_periodic" in kw:
            config.use_periodic = kw["use_periodic"]
        if "periodic_cycle" in kw:
            config.periodic_cycle = kw["periodic_cycle"]
        if "periodic_duration" in kw:
            config.periodic_duration = kw["periodic_duration"]
        if "periodic_reset_momentum" in kw:
            config.periodic_reset_momentum = kw["periodic_reset_momentum"]
        if "eta" in kw:
            config.eta = kw["eta"]
        if "momentum" in kw:
            config.momentum = kw["momentum"]
        if "log_per" in kw:
            config.log_per = kw["log_per"]
        if "ops" in kw:
            config.ops = kw["ops"]
        if "cores" in kw:
            config.cores = kw["cores"]
        if "add_point_weight" in kw:
            config.add_point_weight = kw["add_point_weight"]
        if "update_index_weight" in kw:
            config.update_index_weight = kw["update_index_weight"]
        if "tree_weight" in kw:
            config.tree_weight = kw["tree_weight"]
        if "table_weight" in kw:
            config.table_weight = kw["table_weight"]
        self._config = config



