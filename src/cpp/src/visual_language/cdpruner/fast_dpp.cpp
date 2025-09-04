// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fast_dpp.hpp"
#include "openvino/openvino.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace ov::genai::cdpruner {

FastGreedyDPP::FastGreedyDPP(const Config& config) : m_config(config) {
    // Constructor implementation
}

std::vector<std::vector<size_t>> FastGreedyDPP::select(const ov::Tensor& kernel, size_t num_tokens) {
    // Input validation
    if (kernel.get_shape().size() != 3) {
        throw std::invalid_argument("Kernel must be 3D tensor [B, N, N]");
    }
    
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];
    
    if (shape[1] != shape[2]) {
        throw std::invalid_argument("Kernel matrix must be square [B, N, N]");
    }
    
    if (num_tokens > total_tokens) {
        throw std::invalid_argument("Cannot select more tokens than available");
    }
    
    std::vector<std::vector<size_t>> batch_results(batch_size);
    
    // Process each batch independently
    for (size_t b = 0; b < batch_size; ++b) {
        batch_results[b] = select_single_batch(kernel, b, num_tokens);
    }
    
    return batch_results;
}

std::vector<size_t> FastGreedyDPP::select_single_batch(const ov::Tensor& kernel, size_t batch_idx, size_t num_tokens) {
    auto shape = kernel.get_shape();
    size_t total_tokens = shape[1];
    
    // Initialize working tensors for this batch
    // cis: Orthogonalized vectors [T, N] where T is the number of selected tokens
    ov::Tensor cis(ov::element::f32, {num_tokens, total_tokens});
    
    // di2s: Diagonal elements (marginal gains) [N]
    ov::Tensor di2s(ov::element::f32, {total_tokens});
    
    // Copy diagonal elements from kernel for this batch
    const float* kernel_data = kernel.data<const float>();
    float* di2s_data = di2s.data<float>();
    
    for (size_t i = 0; i < total_tokens; ++i) {
        size_t diag_idx = batch_idx * total_tokens * total_tokens + i * total_tokens + i;
        di2s_data[i] = kernel_data[diag_idx];
    }
    
    std::vector<size_t> selected_indices;
    selected_indices.reserve(num_tokens);
    
    float* cis_data = cis.data<float>();
    std::memset(cis_data, 0, cis.get_byte_size());
    
    // Greedy selection loop - this is the core DPP algorithm
    for (size_t t = 0; t < num_tokens; ++t) {
        // Find the token with maximum marginal gain
        size_t best_idx = argmax(di2s);
        selected_indices.push_back(best_idx);
        
        // Compute the new orthogonalized vector e_i
        // eis = (kernel[batch, best_idx] - sum(cis[:t] * cis[:t, best_idx])) / sqrt(di2s[best_idx])
        update_orthogonal_vector(kernel, batch_idx, best_idx, t, cis, di2s);
        
        // Update marginal gains by subtracting the squared new orthogonal vector
        // di2s -= square(eis)
        update_marginal_gains(t, cis, di2s);

        // Debug output: print cis matrix content
        if (m_config.pruning_debug_mode && t < 10) {
            std::cout << "[CDPruner] === CIS Matrix Content after iteration " << t << " ===" << std::endl;
            std::cout << "[CDPruner] CIS matrix shape: [" << (t+1) << ", " << total_tokens << "]" << std::endl;
            
            const float* cis_data_debug = cis.data<const float>();
            size_t print_tokens = std::min(total_tokens, static_cast<size_t>(10));
            
            // Print each orthogonal vector (each row of cis) - only first 10 elements
            for (size_t row = 0; row <= t; ++row) {
                std::cout << "[CDPruner] cis[" << row << "] (orthogonal vector for selected token " 
                          << selected_indices[row] << "): [";
                
                for (size_t col = 0; col < print_tokens; ++col) {
                    if (col > 0) std::cout << ", ";
                    size_t idx = row * total_tokens + col;
                    std::cout << std::fixed << std::setprecision(4) << cis_data_debug[idx];
                }
                
                if (total_tokens > 10) {
                    std::cout << ", ... (" << (total_tokens - 10) << " more)";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << std::endl;
        }

        // Debug output: print updated conditional kernel matrix after each selection
        if (m_config.pruning_debug_mode && t < 10) {
            // Print current selected indices
            std::cout << "[CDPruner] Selected tokens so far: [";
            for (size_t i = 0; i < selected_indices.size(); ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << selected_indices[i];
            }
            std::cout << "]" << std::endl;

            // Print current marginal gains (di2s) - limited to first 10 elements
            std::cout << "[CDPruner] Current marginal gains: [";
            const float* di2s_data_debug = di2s.data<const float>();
            size_t print_gains_size = std::min(total_tokens, static_cast<size_t>(10));
            
            for (size_t i = 0; i < print_gains_size; ++i) {
                if (i > 0)
                    std::cout << ", ";
                if (di2s_data_debug[i] == -std::numeric_limits<float>::infinity()) {
                    std::cout << "-inf";
                } else {
                    std::cout << std::fixed << std::setprecision(4) << di2s_data_debug[i];
                }
            }
            if (total_tokens > 10) {
                std::cout << ", ... (" << (total_tokens - 10) << " more elements)";
            }
            std::cout << "]" << std::endl << std::endl;
        }

        // Set the selected token's gain to negative infinity to prevent re-selection
        di2s_data[best_idx] = -std::numeric_limits<float>::infinity();
    }
    
    // Sort the selected indices for deterministic output
    std::sort(selected_indices.begin(), selected_indices.end());
    
    return selected_indices;
}

size_t FastGreedyDPP::argmax(const ov::Tensor& scores) {
    const float* data = scores.data<const float>();
    size_t size = scores.get_size();
    
    if (size == 0) {
        throw std::invalid_argument("Cannot find argmax of empty tensor");
    }
    
    size_t best_idx = 0;
    float best_value = -std::numeric_limits<float>::infinity();

    for (size_t i = 1; i < size; ++i) {
        if (data[i] > best_value) {
            best_value = data[i];
            best_idx = i;
        }
    }
    
    return best_idx;
}

void FastGreedyDPP::update_orthogonal_vector(const ov::Tensor& kernel, size_t batch_idx, size_t selected_idx, 
                                           size_t iteration, ov::Tensor& cis, const ov::Tensor& di2s) {
    // This implements the key DPP orthogonalization step:
    // eis = (kernel[batch, selected_idx] - sum(cis[:iteration] * cis[:iteration, selected_idx])) / sqrt(di2s[selected_idx])
    auto kernel_shape = kernel.get_shape();
    size_t total_tokens = kernel_shape[1];
    
    const float* kernel_data = kernel.data<const float>();
    const float* di2s_data = di2s.data<const float>();
    float* cis_data = cis.data<float>();
    // Get the normalization factor
    float norm_factor = std::sqrt(di2s_data[selected_idx] + m_config.numerical_threshold);
    float inv_norm = 1.0f / norm_factor;

    size_t base_kernel_offset = batch_idx * total_tokens * total_tokens + selected_idx * total_tokens;
    const float* kernel_row = kernel_data + base_kernel_offset;

    float* cis_out = cis_data + iteration * total_tokens;

    std::memcpy(cis_out, kernel_row, total_tokens * sizeof(float));

    for (size_t prev_t = 0; prev_t < iteration; ++prev_t) {
        const float* cis_prev_row = cis_data + prev_t * total_tokens;
        float cis_sel = cis_prev_row[selected_idx];

        if (std::abs(cis_sel) < 1e-10f)
            continue;

        for (size_t j = 0; j < total_tokens; ++j) {
            cis_out[j] -= cis_sel * cis_prev_row[j];
        }
    }

    // Process remaining elements
    for (size_t j = 0; j < total_tokens; ++j) {
        cis_out[j] *= inv_norm;
    }
}

void FastGreedyDPP::update_marginal_gains(size_t iteration, const ov::Tensor& cis, ov::Tensor& di2s) {
    // This implements: di2s -= square(eis)
    // where eis is the newly computed orthogonal vector cis[iteration, :]
    
    auto cis_shape = cis.get_shape();
    size_t total_tokens = cis_shape[1];
    
    const float* cis_data = cis.data<const float>();
    float* di2s_data = di2s.data<float>();
    
    // Update marginal gains for all tokens
    for (size_t j = 0; j < total_tokens; ++j) {
        // Skip updating if this token is already selected (marked as negative infinity)
        if (di2s_data[j] == -std::numeric_limits<float>::infinity()) {
            continue;
        }
        
        size_t cis_idx = iteration * total_tokens + j;
        float eis_j = cis_data[cis_idx];
        // Subtract the squared orthogonal component
        if (std::isnan(eis_j)) {
            di2s_data[j] = -std::numeric_limits<float>::max();
            continue;
        }
        di2s_data[j] -= eis_j * eis_j;
    }
}

std::vector<bool> FastGreedyDPP::create_mask(const std::vector<std::vector<size_t>>& selected_indices, 
                                           size_t total_tokens) {
    if (selected_indices.empty()) {
        return std::vector<bool>(total_tokens, false);
    }
    
    size_t batch_size = selected_indices.size();
    std::vector<bool> mask(batch_size * total_tokens, false);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t idx : selected_indices[b]) {
            if (idx < total_tokens) {
                mask[b * total_tokens + idx] = true;
            }
        }
    }
    
    return mask;
}

float FastGreedyDPP::compute_determinant_approximation(const ov::Tensor& kernel, 
                                                     const std::vector<size_t>& selected_indices) {
    // This is a simplified approximation for validation purposes
    // In practice, the greedy algorithm approximates the determinant maximization
    
    if (selected_indices.empty()) {
        return 0.0f;
    }
    
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    
    if (batch_size != 1) {
        throw std::invalid_argument("Determinant approximation only supports single batch");
    }
    
    const float* kernel_data = kernel.data<const float>();
    size_t total_tokens = shape[1];
    
    // Compute the product of diagonal elements of selected tokens as approximation
    float det_approx = 1.0f;
    for (size_t idx : selected_indices) {
        size_t diag_idx = idx * total_tokens + idx;
        det_approx *= kernel_data[diag_idx];
    }
    
    return det_approx;
}

} // namespace ov::genai::cdpruner 