#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>

#if defined _MSC_VER
#include <direct.h>
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include "config.h"
#include "../lib/vptree.h"
#include "../lib/sptree.h"
#include "responsive_tsne.h"

using namespace std;
using namespace panene;

double getEEFactor(Config* conf, int iter) {
    if (conf->use_ee) {
        if (conf->use_periodic) {
            if (iter % conf->periodic_cycle < conf->periodic_duration) return conf->ee_factor;
            return 1.0;
        }

        if (iter < conf->ee_iter) return conf->ee_factor;
    }

    return 1.0;
}
ResponsiveTSNE::ResponsiveTSNE(PyDataSource_* src, bool skip_random, Config* cnf) :
                                skip_random_init(skip_random),
                                conf(cnf),
				no_dims(conf->output_dims),
				K(cnf->perplexity*3),
				source(src),
				sink(Sink(0, K+1)),
				table(Table(
					    source,
					    &sink,
					    K + 1,
					    IndexParams(cnf->num_trees),
					    SearchParams(cnf->num_checks, 0, 0, cnf->cores),
					    TreeWeight(cnf->add_point_weight, cnf->update_index_weight),
					    TableWeight(cnf->tree_weight, cnf->table_weight)

					    )),
				old_ee_factor(1.0f),
				iter(0),
				evalErr(.0),
				momentum(cnf->momentum),
				final_momentum(0.8){
  size_t D = conf->input_dims;
  size_t N = source->size();
  resize_all(N);
  int rand_seed = conf->seed;
  if (skip_random_init != true) {
    if (rand_seed >= 0) {
      //printf("Using random seed: %d\n", rand_seed);
      srand((unsigned int)rand_seed);
    }
    else {
      printf("Using current time as random seed...\n");
      srand((unsigned int)time(NULL));
    }
  }
}

// Perform Responsive t-SNE with Progressive KDTree
void ResponsiveTSNE::resize_all(size_t n){
  size_t cur_n = neighbors.size(); // an aritrary choice
  n = cur_n + n;
  printf("Calling resize_all %d\n", n);
  Y.resize(no_dims*n, 0.);
  dY.resize(no_dims*n, 0.);
  uY.resize(no_dims*n, 0.);
  similarities.resize(n);
  neighbors.resize(n);
  gains.resize(no_dims*n, 1.0);
  sink.resize(n);
  for (auto &tree : table.indexer->trees) {
    tree->capacity = n;
    tree->insertionLog.resize(n);
  }
  table.queued.resize(n);
  size_t N = source->size();
  if(cur_n >= N) return;
  if (skip_random_init != true) {
    //srand(0);
    for (int i = cur_n; i < N * no_dims; i++) Y[i] = randn() * .0001;
    }

}
void ResponsiveTSNE::run_once(vector<int32_t> ids){
  if(ids.size() > 0){
    resize_all(ids.size());
  }
  double perplexity = conf->perplexity;
  double theta = conf->theta;
  size_t D = conf->input_dims;
  double eta = conf->eta;
  size_t N = source->size();
  size_t ops = conf->ops;
  float start_perplex = 0, end_perplex = 0;
  float table_time = 0;
  float ee_factor = getEEFactor(conf, iter);

  if (old_ee_factor != ee_factor) {
    float ratio = ee_factor / old_ee_factor;
    printf("EE changed, ratio = %3f\n", ratio);
    for (auto &sim : similarities) {
      for (auto &kv : sim) {
	kv.second *= ratio;
      }
    }
  }

  old_ee_factor = ee_factor;

  if (table.getSize() < N) {
    updateSimilarity(ee_factor);
  }

  int n = table.getSize();

  computeGradient(n, ee_factor);

  // Update gains
  for (int i = 0; i < n * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
  for (int i = 0; i < n * no_dims; i++) if (gains[i] < .01) gains[i] = .01;

  // Perform gradient update (with momentum and gains)
  for (int i = 0; i < n * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
  for (int i = 0; i < n * no_dims; i++)  Y[i] = Y[i] + uY[i];

  double grad_sum = 0;
  for (int i = 0; i < n * no_dims; i++) {
    grad_sum += dY[i] * dY[i];
  }

  // Make solution zero-mean
  zeroMeanVect(n, no_dims);
  int mom_switch_iter = conf->ee_iter;
  if(iter == mom_switch_iter) {
    momentum = final_momentum;
    printf("switch iter %d", iter);
  }
  //if(iter % 10 == 0){
    evalErr = evaluateError(N, ee_factor);
    //printf("N is %d, error=%lf,  iter = %d\n", N, evalErr, iter);
    //}
  iter++;
}

void ResponsiveTSNE::updateSimilarity(float ee_factor) {

    if (conf->perplexity > K) printf("Perplexity should be lower than K!\n");

    // Update the KNNTable
    UpdateResult ar = table.run(conf->ops);
    /*
       We need to compute val_P for points that are
       1) newly inserted (points in ar.addPointResult)
       2) updated (points in ar.updatePointResult)

       ar.updatedIds has the ids of the updated points.
       The ids of the newly added points can be computed by comparing table.getSize() and ar.addPointResult
       */

       // collect all ids that need to be updated
    vector<size_t> updated;
    vector<double> p(K);
    map<size_t, map<size_t, double>> old;

    for (size_t i = table.getSize() - ar.addPointResult; i < table.getSize(); ++i) {
        // point i has been newly inserted.
        ar.updatedIds.insert(i);

        // for newly added points, we set its initial position to the mean of its neighbors
        std::vector<size_t> indices(K + 1);
        table.getNeighbors(i, indices);

        for (size_t j = i * no_dims; j < (i + 1) * no_dims; ++j) {
            Y[j] = 0;
        }

        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < no_dims; ++j) {
                Y[i * no_dims + j] += Y[indices[k + 1] * no_dims + j] / K;
            }
        }
    }

    for (size_t uid : ar.updatedIds) {
        // the neighbors of uid has been updated
        std::vector<size_t> indices(K + 1);
        std::vector<double> distances(K + 1);

        table.getNeighbors(uid, indices);
        table.getDistances(uid, distances);

        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta = DBL_MAX;
        double tol = 1e-5;

        int iter_ = 0;
        double sum_P;

        // Iterate until we found a good perplexity
        while (!found && iter_ < 200) {

            // Compute Gaussian kernel row
            for (int m = 0; m < K; m++) p[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for (int m = 0; m < K; m++) sum_P += p[m];

            double H = .0;
            for (int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * p[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(conf->perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter_++;
        }

        for (unsigned int m = 0; m < K; m++) p[m] /= sum_P;

        old[uid] = neighbors[uid];
        neighbors[uid].clear();
        for (unsigned int m = 0; m < K; m++) {
            neighbors[uid][indices[m + 1]] = p[m] * ee_factor;
        }
    }

    for (size_t Aid : ar.updatedIds) { // point A
      // neighbors changed, we need to keep the similarities symmetric

      // the neighbors of uid has been updated
        for (auto &it : neighbors[Aid]) { // point B
            size_t Bid = it.first;

            double sAB = it.second;
            double sBA = 0;

            if (neighbors[Bid].count(Aid) > 0)
                sBA = neighbors[Bid][Aid];

            similarities[Aid][Bid] = (sAB + sBA) / 2;
            similarities[Bid][Aid] = (sAB + sBA) / 2;
        }

        for (auto &it : old[Aid]) {
            size_t oldBid = it.first;

            if (neighbors[Aid].count(oldBid) > 0) continue;

            // exit points

            double sAB = 0;
            double sBA = 0;

            if (neighbors[oldBid].count(Aid) > 0)
                sBA = neighbors[oldBid][Aid];

            if (sBA == 0) {
                similarities[Aid].erase(oldBid);
                similarities[oldBid].erase(Aid);
            }
            else {
                similarities[Aid][oldBid] = (sAB + sBA) / 2;
                similarities[oldBid][Aid] = (sAB + sBA) / 2;
            }
        }
    }

    //for(auto &it: similarities[0]) {
      //printf("[%d] %lf\n", it.first, it.second);
    //}
}

void ResponsiveTSNE::run_ids(vector<int32_t> ids, size_t repeat){
  if(ids.size() > 0){
    run_once(ids);
  } else {
    for(int i=0; i < repeat; ++i){
      run_once(ids);
    }
  }
}
// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void ResponsiveTSNE::computeGradient(size_t N, float ee_factor)
{
    // Construct space-partitioning tree on current map
  double* dC = dY.data();
  SPTree* tree = new SPTree(no_dims, &Y[0], N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = (double*)calloc(N * no_dims, sizeof(double));
    double* neg_f = (double*)calloc(N * no_dims, sizeof(double));
    if (pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForces(similarities, N, pos_f, ee_factor);
    for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, conf->theta, neg_f + n * no_dims, &sum_Q);

    // Compute final t-SNE gradient
    for (int i = 0; i < N * no_dims; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Evaluate t-SNE cost function (approximately)
double ResponsiveTSNE::evaluateError(size_t N, float ee_factor)
{
  double theta = conf->theta;
    // Get estimate of normalization term
    SPTree* tree = new SPTree(no_dims, &Y[0], N);
    double* buff = (double*)calloc(no_dims, sizeof(double));
    double sum_Q = .0;
    for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

    // Loop over all edges to compute t-SNE error
    int ind1 = 0, ind2;
    double C = .0, Q;

    double sum_P = .0;
    int j = 0;
    for (auto &p : similarities) {
        if (j >= N) break;
        j++;
        for (auto &it : p) {
            sum_P += it.second;
        }
    }

    sum_P /= ee_factor;

    j = 0;
    for (auto &p : similarities) {
        if (j >= N) break;
        for (auto &it : p) {
            Q = .0;
            ind2 = it.first * no_dims;
            for (int d = 0; d < no_dims; d++) buff[d] = Y[ind1 + d];
            for (int d = 0; d < no_dims; d++) buff[d] -= Y[ind2 + d];
            for (int d = 0; d < no_dims; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += it.second / sum_P * log((it.second / sum_P + FLT_MIN) / (Q + FLT_MIN));
        }
        ind1 += no_dims;
        j++;
    }

    // Clean up memory
    free(buff);
    delete tree;
    return C;
}
void ResponsiveTSNE::dump_Y(){
  for(int i=0; i< Y.size(); ++i){
    printf("%lf ", Y[i]);
  }
}

// Makes data zero-mean
void ResponsiveTSNE::zeroMean(double* X, size_t N, size_t D) {
    // Compute data mean
    double* mean = (double*)calloc(D, sizeof(double));
    if (mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for (int d = 0; d < D; d++) {
        mean[d] /= (double)N;
    }

    // Subtract data mean
    nD = 0;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
	  X[nD + d] -= mean[d];
        }
        nD += D;
    }
    free(mean); mean = NULL;
}


void ResponsiveTSNE::zeroMeanVect(size_t N, size_t D) {
    // Compute data mean
    double* mean = (double*)calloc(D, sizeof(double));
    if (mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            mean[d] += Y[nD + d];
        }
        nD += D;
    }
    for (int d = 0; d < D; d++) {
        mean[d] /= (double)N;
    }

    // Subtract data mean
    nD = 0;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
	  Y[nD + d] -= mean[d];
        }
        nD += D;
    }
    free(mean); mean = NULL;
}


// Generates a Gaussian random number
double ResponsiveTSNE::randn() {
    double x, y, radius;
    do {
        x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
        y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while ((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    y *= radius;
    return x;
}

