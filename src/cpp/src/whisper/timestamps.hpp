// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "whisper.hpp"

namespace ov {
namespace genai {

struct ExtractedSegments {
    std::vector<ov::genai::Segment> segments;
    size_t last_offset;
    std::vector<int64_t> non_timestamp_tokens;
};

ExtractedSegments extract_segments(const std::vector<int64_t>& tokens,
                                   const ov::genai::WhisperGenerationConfig& config,
                                   const size_t nb_max_frames,
                                   const float time_precision);

}  // namespace genai
}  // namespace ov
