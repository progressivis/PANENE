from cpython cimport PyObject
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set
from numpy cimport int64_t, int32_t, uint32_t, float64_t
cimport numpy as np

ctypedef unsigned long size_t

cdef extern from "panene_python.h":
    cdef cppclass SourceABC:
        void set_array(object array)
        void add_to_index(vector[int32_t])
        object get_array() const
        bool is_using_pyarray() const

    cdef cppclass PyDataSource_:
        PyDataSource_(object array)
        void set_array(object array)
        void add_to_index(vector[int32_t])
        
    cdef cppclass PyDataSource:
        PyDataSource(PyDataSource_*)
        void set_array(object array)
        void add_to_index(vector[int32_t])
        object get_array() const
        bool is_using_pyarray() const

    cdef cppclass IndexParams:
        IndexParams()
        int trees

    cdef cppclass SearchParams:
        SearchParams()
        int checks
        float eps
        int sorted
        int cores
   
    cdef cppclass TreeWeight:
        TreeWeight(float, float)

        float addPointWeight
        float updateIndexWeight

    cdef cppclass UpdateResult2:
        UpdateResult2()
        int numPointsInserted
        int addPointOps
        int updateIndexOps
        int addPointResult
        int updateIndexResult
        float addPointElapsed
        float updateIndexElapsed

    cdef cppclass PyNeighbor:
        PyNeighbor()
        int id
        float dist

    cdef cppclass PyResultSet:
        PyResultSet()
        PyResultSet(size_t)
        PyNeighbor operator[](size_t) const
        bint full() const
        size_t k
        float worstDist

    cdef cppclass PyResultSets:
        PyResultSets()
        PyResultSets(size_t)
        PyResultSet operator[](size_t) const
        bint full() const
        size_t size() const

    cdef cppclass Point:
        Point()
        Point(size_t)
        float& operator[](size_t)
        size_t size() const

    cdef cppclass Points:
        Points()
        Points(size_t)
        void reserve(size_t)
        Point& operator[](size_t)
        size_t size() const
        void emplace_back(size_t)

    cdef cppclass IndexABC:
        size_t addPoints(size_t end)
        void beginUpdate()
        UpdateResult2 run(size_t ops) nogil
        void removePoint(size_t id)
        size_t getSize()
        int usedMemory()
        void knnSearch(size_t id, PyResultSet& results, size_t knn, const SearchParams& params) nogil
        void knnSearchVec(const Points& vec, PyResultSets& results, size_t knn, const SearchParams& params) nogil

    cdef cppclass PyIndexL2_:
        PyIndexL2_(PyDataSource_ *ds, IndexParams ip, TreeWeight, float)

    cdef cppclass PyIndexL2:
        PyIndexL2(PyIndexL2_*)
        size_t addPoints(size_t end)
        void beginUpdate()
        UpdateResult2 run(size_t ops) nogil
        void removePoint(size_t id)
        size_t getSize()
        int usedMemory()
        void knnSearch(size_t id, PyResultSet& results, size_t knn, const SearchParams& params) nogil
        void knnSearchVec(const Points& vec, PyResultSets& results, size_t knn, const SearchParams& params) nogil

    cdef cppclass TableWeight:
        TableWeight(float treew, float tablew)
        float treeWeight
        float tableWeight

    cdef cppclass UpdateResult:
        size_t addPointOps
        size_t updateIndexOps
        size_t updateTableOps
        size_t addPointResult
        size_t updateIndexResult
        size_t updateTableResult
        size_t numPointsInserted
        double addPointElapsed
        double updateIndexElapsed
        double updateTableElapsed
        set[size_t] updatedIds

    cdef cppclass PyDataSink:
        PyDataSink(object neighbors, object distances) except +ValueError
        bool is_using_neighbors_pyarray() const
        bool is_using_distances_pyarray() const

    cdef cppclass KNNTableABC:
        size_t getSize()
        UpdateResult run(size_t ops) nogil
        #PyResultSet& getNeighbors(int id)

    cdef cppclass PyKNNTable_:
        PyKNNTable_(PyDataSource_ *ds, PyDataSink *sink, size_t knn, IndexParams ip, SearchParams sp, TreeWeight treew, TableWeight tablew)

    cdef cppclass PyKNNTable:
        PyKNNTable(PyKNNTable_*)
        size_t getSize()
        UpdateResult run(size_t ops) nogil
        #PyResultSet& getNeighbors(int id)

    cdef cppclass ProgressivisSource_:
        ProgressivisSource_(object array)
        
    cdef cppclass ProgressivisSource:
        ProgressivisSource(ProgressivisSource_*)
        void set_array(object array)
        void add_to_index(vector[int32_t])        
        object get_array() const
        bool is_using_pyarray() const

    cdef cppclass PyIndexPvs_:
        PyIndexPvs_(ProgressivisSource_ *ds, IndexParams ip, TreeWeight, float)
        
    cdef cppclass PyIndexPvs:
        PyIndexPvs(PyIndexPvs_*)
        size_t addPoints(size_t end)
        void beginUpdate()
        UpdateResult2 run(size_t ops) nogil
        void removePoint(size_t id)
        size_t getSize()
        int usedMemory()
        void knnSearch(size_t id, PyResultSet& results, size_t knn, const SearchParams& params) nogil
        void knnSearchVec(const Points& vec, PyResultSets& results, size_t knn, const SearchParams& params) nogil
        
    cdef cppclass PyKNNTablePvs_:
        PyKNNTablePvs_(ProgressivisSource_ *ds, PyDataSink *sink, size_t knn, IndexParams ip, SearchParams sp, TreeWeight treew, TableWeight tablew)
    cdef cppclass PyKNNTablePvs:
        PyKNNTablePvs(PyKNNTablePvs_*)
        size_t getSize()
        UpdateResult run(size_t ops) nogil
        #PyResultSet& getNeighbors(int id)

cdef extern from "config.h":
    cdef cppclass Config:
        size_t n
        size_t input_dims
        size_t output_dims
        double theta
        double perplexity
        double eta
        double momentum
        size_t max_iter

        bool use_ee  # early exaggeration
        double ee_factor
        size_t ee_iter
        int seed

        bool use_periodic
        size_t periodic_cycle
        size_t periodic_duration

        bool periodic_reset_momentum
        size_t log_per

        size_t ops
        size_t cores

        size_t num_trees
        size_t num_checks

        float add_point_weight
        float update_index_weight
        float tree_weight
        float table_weight

cdef extern from "responsive_tsne.h":
    cdef cppclass ResponsiveTSNE:
        vector[double] Y
        double evalErr
        ResponsiveTSNE(PyDataSource_* source, bool skip_random, Config* cnf)
        void run_ids(vector[int32_t], size_t)
        void dump_Y()

cdef extern from "responsive_tsne.cpp":
    pass


cdef extern from "../lib/sptree.cpp":
    pass
    
