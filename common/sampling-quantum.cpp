#include "sampling-quantum.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static constexpr uint64_t QUANTUM_FLOOR_M = 1ULL << 32;
static constexpr uint64_t QUANTUM_FLOOR_MASK = QUANTUM_FLOOR_M - 1;
static constexpr int QUANTUM_FLOOR_K_MAX = 1 << 20;

std::vector<uint32_t> quantum_floor_compute_G_rows() {
    std::vector<uint32_t> rows(32);
    for (int i = 0; i < 32; ++i) {
        uint32_t row = 0;
        for (int j = 0; j < 32; ++j) {
            const int I_ij = i == j ? 1 : 0;
            const int H_ij = __builtin_popcount((unsigned) (i & j)) & 1;
            const int J_ij = 1;
            const int g_ij = I_ij ^ H_ij ^ J_ij;
            if (g_ij) {
                row |= 1u << j;
            }
        }
        rows[i] = row;
    }
    return rows;
}

uint32_t quantum_floor_spread(uint32_t u, const std::vector<uint32_t> & rows) {
    if (rows.size() != 32) {
        throw std::runtime_error("quantum_floor: spreader requires 32 row masks");
    }

    uint32_t y = 0;
    for (int i = 0; i < 32; ++i) {
        if (__builtin_parity(u & rows[i])) {
            y |= 1u << i;
        }
    }
    return y;
}

quantum_floor_allocation quantum_floor_build_allocation(const std::vector<double> & probs, uint32_t K) {
    if (probs.empty()) {
        throw std::runtime_error("quantum_floor: empty probability vector");
    }
    if ((uint64_t) K * probs.size() > QUANTUM_FLOOR_M) {
        throw std::runtime_error("quantum_floor: K is too large for vocabulary size");
    }

    quantum_floor_allocation result;
    result.slots.resize(probs.size());
    result.cdf.resize(probs.size());

    std::vector<bool> floored(probs.size(), false);
    uint64_t total = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        const double p = std::isfinite(probs[i]) && probs[i] > 0.0 ? probs[i] : 0.0;
        const auto slots = (uint64_t) std::floor(p * (double) QUANTUM_FLOOR_M);
        result.slots[i] = slots;
        total += slots;
    }

    for (size_t i = 0; i < result.slots.size(); ++i) {
        if (result.slots[i] < K) {
            total += (uint64_t) K - result.slots[i];
            result.slots[i] = K;
            floored[i] = true;
        }
    }

    uint64_t resolved_total = 0;
    size_t largest_resolved = result.slots.size();
    for (size_t i = 0; i < result.slots.size(); ++i) {
        if (!floored[i]) {
            resolved_total += result.slots[i];
            if (largest_resolved == result.slots.size() || result.slots[i] > result.slots[largest_resolved]) {
                largest_resolved = i;
            }
        }
    }

    if (resolved_total == 0 && total != QUANTUM_FLOOR_M) {
        throw std::runtime_error("quantum_floor: no resolved tokens available for floor compensation");
    }

    const int64_t delta = (int64_t) total - (int64_t) QUANTUM_FLOOR_M;
    if (delta != 0) {
        uint64_t adjusted_total = 0;
        for (size_t i = 0; i < result.slots.size(); ++i) {
            if (!floored[i]) {
                const long double share = (long double) std::llabs(delta) * (long double) result.slots[i] / (long double) resolved_total;
                const uint64_t adj = (uint64_t) std::llround(share);
                if (delta > 0) {
                    result.slots[i] = adj > result.slots[i] ? 0 : result.slots[i] - adj;
                } else {
                    result.slots[i] += adj;
                }
            }
            adjusted_total += result.slots[i];
        }
        total = adjusted_total;
    }

    while (total != QUANTUM_FLOOR_M) {
        if (largest_resolved == result.slots.size()) {
            throw std::runtime_error("quantum_floor: unable to correct slot residual");
        }
        if (total > QUANTUM_FLOOR_M) {
            if (result.slots[largest_resolved] == 0) {
                throw std::runtime_error("quantum_floor: slot residual underflow");
            }
            --result.slots[largest_resolved];
            --total;
        } else {
            ++result.slots[largest_resolved];
            ++total;
        }
    }

    uint64_t c = 0;
    for (size_t i = 0; i < result.slots.size(); ++i) {
        c += result.slots[i];
        result.cdf[i] = c;
    }
    result.cdf.back() = QUANTUM_FLOOR_MASK;

    return result;
}

struct llama_sampler_quantum_floor {
    quantum_floor_params params;
    int32_t n_vocab;
    std::vector<uint32_t> rows;

    bool tax_logged = false;
    bool can_prefetch = false;
    std::future<uint32_t> prefetched_word;

    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> words_produced{0};
    std::deque<uint8_t> bit_window;
    uint64_t bit_ones = 0;
    std::chrono::steady_clock::time_point last_diag_log = std::chrono::steady_clock::now();

    std::mutex log_mutex;
    std::ofstream log_file;

#ifndef _WIN32
    int sock = -1;
#endif
};

static void quantum_floor_log(llama_sampler_quantum_floor * ctx, const char * level, const std::string & msg) {
    const std::string line = std::string("quantum_floor ") + level + ": " + msg;
    if (!ctx->params.log_path.empty()) {
        std::lock_guard<std::mutex> lock(ctx->log_mutex);
        if (!ctx->log_file.is_open()) {
            ctx->log_file.open(ctx->params.log_path, std::ios::app);
        }
        if (ctx->log_file.is_open()) {
            ctx->log_file << line << '\n';
            ctx->log_file.flush();
            return;
        }
    }
    if (std::strcmp(level, "WARN") == 0) {
        LOG_WRN("%s\n", line.c_str());
    } else {
        LOG_INF("%s\n", line.c_str());
    }
}

static void quantum_floor_health_probe(llama_sampler_quantum_floor * ctx) {
    try {
        httplib::Client cli(ctx->params.qrng_host, 5001);
        cli.set_connection_timeout(1, 0);
        cli.set_read_timeout(1, 0);
        cli.set_write_timeout(1, 0);
        auto res = cli.Get("/health");
        if (!res || res->status != 200) {
            quantum_floor_log(ctx, "WARN", "health probe failed; proceeding to TCP entropy socket");
            return;
        }
        try {
            const auto data = nlohmann::ordered_json::parse(res->body);
            std::ostringstream ss;
            ss << "health probe counters:";
            bool found = false;
            for (auto it = data.begin(); it != data.end(); ++it) {
                std::string key = it.key();
                std::string key_lower = key;
                std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), [](unsigned char c) {
                    return (char) std::tolower(c);
                });
                const bool interesting = key_lower.find("cnt") != std::string::npos ||
                    key_lower.find("byte") != std::string::npos ||
                    key_lower.find("rate") != std::string::npos ||
                    key_lower.find("word") != std::string::npos;
                if (!interesting || !(it->is_number() || it->is_string() || it->is_boolean())) {
                    continue;
                }
                ss << ' ' << key << '=' << it->dump();
                found = true;
            }
            quantum_floor_log(ctx, "INFO", found ? ss.str() : "health probe JSON: " + data.dump());
        } catch (const std::exception &) {
            quantum_floor_log(ctx, "INFO", "health probe body: " + res->body);
        }
    } catch (const std::exception & e) {
        quantum_floor_log(ctx, "WARN", std::string("health probe failed: ") + e.what() + "; proceeding to TCP entropy socket");
    }
}

#ifndef _WIN32
static int quantum_floor_connect_tcp(const std::string & host, int port) {
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo * res = nullptr;
    const std::string port_str = std::to_string(port);
    const int rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
    if (rc != 0) {
        throw std::runtime_error(std::string("quantum_floor: getaddrinfo failed: ") + gai_strerror(rc));
    }

    int fd = -1;
    for (auto * rp = res; rp != nullptr; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd == -1) {
            continue;
        }
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd == -1) {
        throw std::runtime_error("quantum_floor: failed to connect to QRNG entropy socket");
    }
    return fd;
}

static void quantum_floor_note_byte(llama_sampler_quantum_floor * ctx, uint8_t byte) {
    for (int i = 0; i < 8; ++i) {
        const uint8_t bit = (byte >> i) & 1;
        ctx->bit_window.push_back(bit);
        ctx->bit_ones += bit;
        if (ctx->bit_window.size() > 65536) {
            ctx->bit_ones -= ctx->bit_window.front();
            ctx->bit_window.pop_front();
        }
    }
}

static void quantum_floor_log_diagnostics_if_due(llama_sampler_quantum_floor * ctx) {
    const auto now = std::chrono::steady_clock::now();
    if (now - ctx->last_diag_log < std::chrono::seconds(1)) {
        return;
    }

    double mean = 0.0;
    double lag1_corr = 0.0;
    if (!ctx->bit_window.empty()) {
        mean = (double) ctx->bit_ones / (double) ctx->bit_window.size();
    }
    if (ctx->bit_window.size() > 1) {
        double xy = 0.0;
        double x = 0.0;
        double y = 0.0;
        for (size_t i = 1; i < ctx->bit_window.size(); ++i) {
            x += ctx->bit_window[i - 1];
            y += ctx->bit_window[i];
            xy += ctx->bit_window[i - 1] & ctx->bit_window[i];
        }
        const double n = (double) (ctx->bit_window.size() - 1);
        const double mx = x / n;
        const double my = y / n;
        const double cov = xy / n - mx * my;
        const double vx = mx * (1.0 - mx);
        const double vy = my * (1.0 - my);
        if (vx > 0.0 && vy > 0.0) {
            lag1_corr = cov / std::sqrt(vx * vy);
        }
    }

    quantum_floor_log(ctx, "INFO", string_format("bytes=%" PRIu64 " words=%" PRIu64 " mean=%.6f lag1_corr=%.6f",
                ctx->bytes_received.load(), ctx->words_produced.load(), mean, lag1_corr));
    ctx->last_diag_log = now;
}

static uint32_t quantum_floor_read_word(llama_sampler_quantum_floor * ctx) {
    uint8_t bytes[4] {};
    size_t n_read = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ctx->params.recv_timeout_ms);

    while (n_read < sizeof(bytes)) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            throw std::runtime_error("quantum_floor: timed out waiting for QRNG entropy");
        }

        const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        struct pollfd pfd {};
        pfd.fd = ctx->sock;
        pfd.events = POLLIN;

        const int prc = poll(&pfd, 1, std::max<int>(1, (int) remaining_ms));
        if (prc < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("quantum_floor: poll failed: ") + strerror(errno));
        }
        if (prc == 0) {
            throw std::runtime_error("quantum_floor: timed out waiting for QRNG entropy");
        }

        const ssize_t n = recv(ctx->sock, bytes + n_read, sizeof(bytes) - n_read, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("quantum_floor: recv failed: ") + strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error("quantum_floor: QRNG socket closed");
        }

        for (ssize_t i = 0; i < n; ++i) {
            quantum_floor_note_byte(ctx, bytes[n_read + i]);
        }
        n_read += (size_t) n;
        ctx->bytes_received += (uint64_t) n;
    }

    ctx->words_produced++;
    quantum_floor_log_diagnostics_if_due(ctx);

    const uint32_t word = ((uint32_t) bytes[3] << 24) | ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[1] << 8) | (uint32_t) bytes[0];
    return word;
}
#endif

static void quantum_floor_start_prefetch(llama_sampler_quantum_floor * ctx) {
#ifndef _WIN32
    if (!ctx->can_prefetch || ctx->prefetched_word.valid()) {
        return;
    }

    ctx->prefetched_word = std::async(std::launch::async, [ctx] {
        return quantum_floor_read_word(ctx);
    });
#else
    (void) ctx;
#endif
}

static uint32_t quantum_floor_next_word(llama_sampler_quantum_floor * ctx) {
#ifdef _WIN32
    throw std::runtime_error("quantum_floor: QRNG TCP sampler is not implemented on Windows");
#else
    if (ctx->prefetched_word.valid()) {
        return ctx->prefetched_word.get();
    }
    return quantum_floor_read_word(ctx);
#endif
}

static const char * llama_sampler_quantum_floor_name(const llama_sampler *) {
    return "quantum_floor";
}

static void llama_sampler_quantum_floor_accept(llama_sampler * smpl, llama_token) {
    auto * ctx = (llama_sampler_quantum_floor *) smpl->ctx;
    quantum_floor_start_prefetch(ctx);
}

static void llama_sampler_quantum_floor_reset(llama_sampler * smpl) {
    auto * ctx = (llama_sampler_quantum_floor *) smpl->ctx;
    ctx->tax_logged = false;
    ctx->can_prefetch = false;
    if (ctx->prefetched_word.valid()) {
        try {
            (void) ctx->prefetched_word.get();
        } catch (const std::exception &) {
        }
    }
}

static void quantum_floor_log_tax(llama_sampler_quantum_floor * ctx, const std::vector<double> & probs) {
    uint64_t under_resolved_count = 0;
    double S_K = 0.0;

    for (size_t i = 0; i < probs.size(); ++i) {
        const auto slots = (uint64_t) std::floor(probs[i] * (double) QUANTUM_FLOOR_M);
        if (slots < (uint32_t) ctx->params.K) {
            ++under_resolved_count;
            S_K += probs[i];
        }
    }

    const double F_K = (double) ((uint64_t) ctx->params.K * under_resolved_count) / (double) QUANTUM_FLOOR_M;
    const double delta_K = F_K - S_K;
    const double aggregate_boost = S_K > 0.0 ? F_K / S_K : INFINITY;

    quantum_floor_log(ctx, "INFO", "diagnostic (first sample):");
    quantum_floor_log(ctx, "INFO", string_format("  K                       = %d", ctx->params.K));
    quantum_floor_log(ctx, "INFO", string_format("  n_vocab                 = %zu", probs.size()));
    quantum_floor_log(ctx, "INFO", string_format("  under-resolved tokens   = %" PRIu64 " (%.2f%% of vocab)",
                under_resolved_count, 100.0 * (double) under_resolved_count / (double) probs.size()));
    quantum_floor_log(ctx, "INFO", string_format("  original tail mass S_K  = %.6f%% (%.3e)", S_K * 100.0, S_K));
    quantum_floor_log(ctx, "INFO", string_format("  floored tail mass F_K   = %.6f%% (%.3e)", F_K * 100.0, F_K));
    quantum_floor_log(ctx, "INFO", string_format("  multiverse tax \xce\x94_K      = %.6f%% (%.3e)", delta_K * 100.0, delta_K));
    quantum_floor_log(ctx, "INFO", string_format("  aggregate tail boost    = %.2fx", aggregate_boost));
}

static void llama_sampler_quantum_floor_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_quantum_floor *) smpl->ctx;

    if (cur_p->size == 0) {
        cur_p->selected = -1;
        return;
    }
    if (ctx->params.require_full_vocab) {
        if ((int32_t) cur_p->size != ctx->n_vocab) {
            throw std::runtime_error("quantum_floor sampler requires full vocabulary; remove truncation samplers from the chain");
        }
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (cur_p->data[i].id != (llama_token) i || !std::isfinite(cur_p->data[i].logit)) {
                throw std::runtime_error("quantum_floor sampler requires full vocabulary; remove truncation samplers from the chain");
            }
        }
    }

    const uint32_t raw = quantum_floor_next_word(ctx);
    ctx->can_prefetch = true;
    const uint32_t u = quantum_floor_spread(raw, ctx->rows);

    std::vector<double> probs(cur_p->size);
    double p_sum = 0.0;
    bool has_probs = false;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p != 0.0f) {
            has_probs = true;
        }
        p_sum += cur_p->data[i].p;
    }

    if (has_probs && std::isfinite(p_sum) && p_sum > 0.0) {
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (ctx->params.require_full_vocab && cur_p->data[i].p <= 0.0f) {
                throw std::runtime_error("quantum_floor sampler requires full vocabulary; remove truncation samplers from the chain");
            }
            probs[i] = cur_p->data[i].p / p_sum;
            cur_p->data[i].p = (float) probs[i];
        }
    } else {
        float max_l = cur_p->data[0].logit;
        for (size_t i = 1; i < cur_p->size; ++i) {
            max_l = std::max(max_l, cur_p->data[i].logit);
        }

        double sum = 0.0;
        for (size_t i = 0; i < cur_p->size; ++i) {
            const double p = std::exp((double) cur_p->data[i].logit - (double) max_l);
            probs[i] = p;
            sum += p;
        }
        if (!(sum > 0.0) || !std::isfinite(sum)) {
            throw std::runtime_error("quantum_floor: invalid probability mass");
        }
        for (size_t i = 0; i < cur_p->size; ++i) {
            probs[i] /= sum;
            cur_p->data[i].p = (float) probs[i];
        }
    }

    if (ctx->params.debug_tax && !ctx->tax_logged) {
        quantum_floor_log_tax(ctx, probs);
        ctx->tax_logged = true;
    }

    const auto alloc = quantum_floor_build_allocation(probs, (uint32_t) ctx->params.K);
    const auto it = std::upper_bound(alloc.cdf.begin(), alloc.cdf.end(), (uint64_t) u);
    cur_p->selected = it == alloc.cdf.end() ? cur_p->size - 1 : (int32_t) std::distance(alloc.cdf.begin(), it);
}

static llama_sampler * llama_sampler_quantum_floor_clone(const llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_quantum_floor *) smpl->ctx;
    return llama_sampler_init_quantum_floor(ctx->params, ctx->n_vocab);
}

static void llama_sampler_quantum_floor_free(llama_sampler * smpl) {
    auto * ctx = (llama_sampler_quantum_floor *) smpl->ctx;
#ifndef _WIN32
    if (ctx->sock != -1) {
        shutdown(ctx->sock, SHUT_RDWR);
    }
#endif
    if (ctx->prefetched_word.valid()) {
        try {
            (void) ctx->prefetched_word.get();
        } catch (const std::exception &) {
        }
    }
#ifndef _WIN32
    if (ctx->sock != -1) {
        close(ctx->sock);
    }
#endif
    delete ctx;
}

static llama_sampler_i llama_sampler_quantum_floor_i = {
    /* .name              = */ llama_sampler_quantum_floor_name,
    /* .accept            = */ llama_sampler_quantum_floor_accept,
    /* .apply             = */ llama_sampler_quantum_floor_apply,
    /* .reset             = */ llama_sampler_quantum_floor_reset,
    /* .clone             = */ llama_sampler_quantum_floor_clone,
    /* .free              = */ llama_sampler_quantum_floor_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

llama_sampler * llama_sampler_init_quantum_floor(const quantum_floor_params & params, int32_t n_vocab) {
    if (params.qrng_host.empty()) {
        throw std::runtime_error("quantum_floor: qrng_host is empty");
    }
    if (params.qrng_port <= 0 || params.qrng_port > 65535) {
        throw std::runtime_error("quantum_floor: invalid QRNG port");
    }
    if (params.K < 1) {
        throw std::runtime_error("quantum_floor: K must be >= 1");
    }
    if (params.K > QUANTUM_FLOOR_K_MAX) {
        throw std::runtime_error("quantum_floor: K must be <= 2^20; current value would consume most of the address space");
    }
    const uint64_t floor_reserved = (uint64_t) params.K * (uint64_t) n_vocab;
    if (floor_reserved > QUANTUM_FLOOR_M) {
        throw std::runtime_error("quantum_floor: K * n_vocab exceeds the 2^32 address space");
    }
    if (params.recv_timeout_ms <= 0) {
        throw std::runtime_error("quantum_floor: recv_timeout_ms must be positive");
    }

    auto * ctx = new llama_sampler_quantum_floor();
    ctx->params = params;
    ctx->n_vocab = n_vocab;
    ctx->rows = quantum_floor_compute_G_rows();

    if (floor_reserved > (1ULL << 31)) {
        const double pct = 100.0 * (double) floor_reserved / (double) QUANTUM_FLOOR_M;
        quantum_floor_log(ctx, "WARN", string_format("K=%d with n_vocab=%d reserves %.3f%% of address space for the floor; resolved tokens will be compressed accordingly",
                    params.K, n_vocab, pct));
    }

    quantum_floor_health_probe(ctx);

#ifdef _WIN32
    delete ctx;
    throw std::runtime_error("quantum_floor: QRNG TCP sampler is not implemented on Windows");
#else
    try {
        ctx->sock = quantum_floor_connect_tcp(params.qrng_host, params.qrng_port);
        quantum_floor_log(ctx, "INFO", string_format("connected to QRNG Lever stream at %s:%d", params.qrng_host.c_str(), params.qrng_port));
    } catch (...) {
        delete ctx;
        throw;
    }
#endif

    return llama_sampler_init(&llama_sampler_quantum_floor_i, ctx);
}
