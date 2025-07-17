// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "structured_output_controller.hpp"

namespace ov {
namespace genai {

std::unordered_map<std::string, StructuredOutputController::BackendFactory>&
StructuredOutputController::get_backend_registry() {
    static std::unordered_map<std::string, BackendFactory> registry;
    return registry;
}

std::string& StructuredOutputController::get_default_backend_name() {
    static std::string default_backend = "xgrammar";
    return default_backend;
}

void StructuredOutputController::register_backend(const std::string& name, BackendFactory factory) {
    get_backend_registry()[name] = std::move(factory);
}

void StructuredOutputController::set_default_backend(const std::string& name) {
    if (get_backend_registry().find(name) == get_backend_registry().end()) {
        OPENVINO_THROW("Cannot set default backend to unregistered backend: " + name);
    }

    get_default_backend_name() = name;
}

StructuredOutputController::StructuredOutputController(const ov::genai::Tokenizer& tokenizer,
                                                       std::optional<int> vocab_size)
    : m_tokenizer(tokenizer), m_vocab_size(vocab_size) {}

std::shared_ptr<LogitTransformers::ILogitTransformer>
StructuredOutputController::get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters) {
    auto& guided_gen_config = sampling_parameters.structured_output_config;
    if (!guided_gen_config.has_value()) {
        OPENVINO_THROW("Structured output is not enabled in the provided GenerationConfig.");
    }
    std::string backend_name = (*guided_gen_config).backend.value_or(get_default_backend_name());
    
    std::unique_lock<std::mutex> lock(m_mutex);

    // Check if backend already instantiated
    auto impl_it = m_impls.find(backend_name);
    if (impl_it == m_impls.end()) {
        // Backend not instantiated yet, create it
        auto& registry = get_backend_registry();
        auto factory_it = registry.find(backend_name);
        if (factory_it == registry.end()) {
            OPENVINO_THROW("Structured output backend not found: " + backend_name);
        }

        // Create the backend instance and store it
        const auto start = std::chrono::steady_clock::now();
        m_impls[backend_name] = factory_it->second(m_tokenizer, m_vocab_size);
        impl_it = m_impls.find(backend_name);
        const auto end = std::chrono::steady_clock::now();
        m_init_grammar_compiler_times[backend_name] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    // Use the instantiated backend
    const auto start = std::chrono::steady_clock::now();
    auto res = impl_it->second->get_logits_transformer(sampling_parameters);
    const auto end = std::chrono::steady_clock::now();
    m_grammar_compile_times.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return res;
}

std::pair<std::map<std::string, float>, std::vector<float>> StructuredOutputController::get_times() const {
    return {m_init_grammar_compiler_times, m_grammar_compile_times};
}

void StructuredOutputController::clear_compile_times() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_grammar_compile_times.clear();
}

} // namespace genai
} // namespace ov
