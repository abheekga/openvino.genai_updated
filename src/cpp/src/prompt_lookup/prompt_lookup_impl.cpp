// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <thread>

#include "prompt_lookup_impl.hpp"
#include "text_callback_streamer.hpp"

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

GenerationHandle
ContinuousBatchingPipeline::PromptLookupImpl::add_request(uint64_t request_id,
                                                          const ov::Tensor& input_ids,
                                                          ov::genai::GenerationConfig sampling_params) {
    OPENVINO_ASSERT(sampling_params.is_prompt_lookup(), "`max_ngram_size` && `num_assistant_tokens` should be specified for `prompt lookup decoding`");
    return m_pipeline->add_request(request_id, input_ids, sampling_params);
};

GenerationHandle
ContinuousBatchingPipeline::PromptLookupImpl::add_request(uint64_t request_id,
                                                          const std::string& prompt,
                                                          ov::genai::GenerationConfig sampling_params) {
    OPENVINO_ASSERT(sampling_params.is_prompt_lookup(), "`max_ngram_size` && `num_assistant_tokens` should be specified for `prompt lookup decoding`");
    return m_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::PromptLookupImpl::has_non_finished_requests() {
    return m_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::PromptLookupImpl::step() {
    auto& raw_perf_counters = m_perf_metrics.raw_metrics;

    ManualTimer step_timer("prompt_lookup_decoding: step()");
    step_timer.start();

    ManualTimer candidates_timer("prompt_lookup_decoding: generate_candidates()");
    candidates_timer.start();
    m_pipeline->generate_candidates();
    candidates_timer.end();
    m_sd_metrics.draft_duration += candidates_timer.get_duration();
    auto generated_len_before = m_pipeline->get_generated_request_len();

    ManualTimer main_timer("prompt_lookup_decoding: pipeline: step()");
    main_timer.start();
    m_pipeline->step();
    main_timer.end();
    m_sd_metrics.main_duration += main_timer.get_duration();
    m_pipeline_metrics = m_pipeline->get_metrics();
    auto generated_len_after = m_pipeline->get_generated_request_len();

    for (const auto request : generated_len_before) {
        auto request_id = request.first;
        auto prev_validation_len = request.second.second;
        if (prev_validation_len == 0) {
            continue;
        }
        size_t num_matches = prev_validation_len;
        float acceptance_rate = 1.f;
        if (generated_len_after.count(request.first)) {
            auto present_req_len = generated_len_after.at(request.first).first;
            auto prev_full_req_len = request.second.first;

            num_matches = (present_req_len - prev_full_req_len - 1);
            acceptance_rate = static_cast<float>(num_matches) / static_cast<float>(prev_validation_len);
        }
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate * 100);
        m_sd_metrics.update_draft_accepted_tokens(request_id, num_matches);
    }

    // update perf metrics
    const auto num_generated_tokens = m_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);

        auto infer_duration = step_timer.get_duration_microsec();

        raw_perf_counters.m_token_infer_durations.emplace_back(infer_duration);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_duration);
        raw_perf_counters.m_new_token_times.emplace_back(main_timer.get_end_time());
    }

    if (generated_len_after.empty() && 0) {
        m_sd_metrics.print(true);
        m_sd_metrics.clean_up();
    }
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::PromptLookupImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                       const std::vector<GenerationConfig>& sampling_params,
                                                       const StreamerVariant& streamer) {
    m_perf_metrics = PerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};

    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    ManualTimer generate_timer("speculative_decoding: generate()");
    generate_timer.start();

    // checks that all requests has the same LoRA adapters property value
    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
            "LoRA adapters value must be the same for all requests");
    }
    m_pipeline->set_adapters(sampling_params[0].adapters);

    const std::shared_ptr<StreamerBase>& streamer_ptr = std::visit(overloaded{
        [](std::monostate) -> std::shared_ptr<StreamerBase> {
            return nullptr;
        },
        [](const std::shared_ptr<StreamerBase>& streamer) {
            return streamer;
        },
        [this](const std::function<bool(std::string)>& streamer) -> std::shared_ptr<StreamerBase> {
            return std::make_unique<TextCallbackStreamer>(m_tokenizer, streamer);
        }
    }, streamer);

    OPENVINO_ASSERT(streamer_ptr == nullptr || input_ids.size() == 1 && (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        OPENVINO_ASSERT(sampling_params[request_id].is_prompt_lookup(), "`max_ngram_size` && `num_assistant_tokens` should be specified for `prompt lookup decoding`");
        generations.push_back(m_pipeline->add_request(request_id, input_ids[request_id], sampling_params[request_id]));
    }
    auto all_requests = m_pipeline->get_awaiting_requests();

    std::atomic<bool> has_active_requests = has_non_finished_requests();
    auto& generation = generations.at(0);

    // create variables to make optimal thread-safe streaming
    std::mutex mutex;
    std::unique_lock lock(mutex);
    std::condition_variable cv;

    // to define streaming thread
    std::shared_ptr<std::thread> t_stream_ptr = nullptr;
    if (streamer_ptr) {
        // define stream token lambda to use in `t_stream_ptr`
        auto stream_tokens = [this, &generation, &streamer_ptr, &has_active_requests, &cv, &lock]() {
            while (has_active_requests || generation->can_read()) {
                // waiting for any tokens or request finishing
                cv.wait(lock, [&generation, &has_active_requests]{
                    return generation->can_read() || !has_active_requests;
                });

                if (generation->can_read()) {
                    std::unordered_map<uint64_t, GenerationOutput> token = generation->read();
                    for (const auto& gen_token : token.begin()->second.generated_ids) {
                        if (streamer_ptr->put(gen_token)) {
                            generation->drop();
                            break;
                        }
                    }
                }
            };
            streamer_ptr->end();
        };

        // to define streaming thread
        t_stream_ptr = std::make_shared<std::thread>([&stream_tokens] {
            stream_tokens();
        });
    }

    std::exception_ptr thrown_exception = nullptr;
    while (has_active_requests) {
        try {
            step();
        } catch (...) {
            drop_requests(); // remove all requests from pipeline state in case of exception
            thrown_exception = std::current_exception();
        }
        has_active_requests = has_non_finished_requests();
        cv.notify_one();
        if (thrown_exception) {
            throw thrown_exception;
        }
    }

    // waiting for competion of streaming
    if (t_stream_ptr && t_stream_ptr->joinable()) {
        t_stream_ptr->join();
    }

    OPENVINO_ASSERT(m_pipeline->is_requests_empty(), "Internal error: current request is supposed to be dropped within step() function as completed");

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto sampling_params = request->get_sampling_parameters();
        const auto& sequences = request->get_finished_sequences();
        size_t num_outputs = std::min(sampling_params.num_return_sequences, sequences.size());

        EncodedGenerationResult result;
        result.m_request_id = request_id;
        result.m_generation_ids.resize(num_outputs);
        result.m_scores.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            const auto & sequence = sequences[i];
            const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params) : sequence->get_cumulative_log_prob();
            const auto & generated_ids = sequence->get_generated_ids();

            if (sampling_params.echo) {
                result.m_generation_ids[i] = request->get_prompt_ids();
            }
            std::copy(generated_ids.begin(), generated_ids.end(), std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
        }

        result.m_status = generations[request_id]->get_status();

        // The same perf metrics for each sequence, only tokenization/detokenization will differ.
        m_perf_metrics.raw_metrics.generate_durations.clear();
        m_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());
        m_perf_metrics.num_input_tokens = request->get_prompt_len();
        m_perf_metrics.evaluate_statistics(generate_timer.get_start_time());

        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    generate_timer.end();
    return results;
}

SpeculativeDecodingMetrics
ContinuousBatchingPipeline::PromptLookupImpl::get_metrics() {
    return m_sd_metrics;
};

void ContinuousBatchingPipeline::PromptLookupImpl::drop_requests() {
    m_pipeline->drop_requests();
}
}
