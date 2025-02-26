// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderPhi3V : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderPhi3V : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderPhi3V(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config
    );

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

private:
    ov::InferRequest m_hd_feature_transformer;
    ov::InferRequest m_vision_projection;
    std::vector<size_t> m_tokens_per_images;
};

} // namespace ov::genai
