#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct quantum_floor_params {
    std::string qrng_host;
    int         qrng_port = 5003;
    int         K = 64;
    int         buffer_size = 256;
    int         recv_timeout_ms = 2000;
    bool        require_full_vocab = true;
    bool        debug_tax = false;
    std::string log_path;
};

struct quantum_floor_allocation {
    std::vector<uint64_t> slots;
    std::vector<uint64_t> cdf;
};

llama_sampler * llama_sampler_init_quantum_floor(const quantum_floor_params & params, int32_t n_vocab);

std::vector<uint32_t> quantum_floor_compute_G_rows();
uint32_t quantum_floor_spread(uint32_t u, const std::vector<uint32_t> & rows);
quantum_floor_allocation quantum_floor_build_allocation(const std::vector<double> & probs, uint32_t K);
