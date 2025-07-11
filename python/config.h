#pragma once

struct Config {
    size_t n = 0;
    size_t input_dims = 0;
    size_t output_dims = 0;
    double theta = 0.5;
    double perplexity = 10;
    double eta = 200;
    double momentum = .5;
    size_t max_iter = 1000;
    
    bool use_ee = true; // early exaggeration
    double ee_factor = 12.0;
    size_t ee_iter = 250;   
    int seed = -1;

    bool use_periodic = false;
    size_t periodic_cycle = 100;
    size_t periodic_duration = 30;
    
    bool periodic_reset_momentum = false;
    size_t log_per = 10;

    size_t ops = 300;
    size_t cores = 4;

    size_t num_trees = 4;
    size_t num_checks = 1024;

    float add_point_weight = 0.7f;
    float update_index_weight = 0.3f;
    float tree_weight = 0.5f;
    float table_weight = 0.5f;
};
