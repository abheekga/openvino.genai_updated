# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_JS_DIR
from test_utils import run_sample

class TestTextEmbeddingPipeline:
    @pytest.mark.rag
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["BAAI/bge-small-en-v1.5"], indirect=True)
    def test_sample_text_embedding_pipeline(self, convert_model):
        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "rag/text_embeddings.py")
        py_command = [sys.executable, py_script, convert_model, "Document 1", "Document 2"]
        py_result = run_sample(py_command)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, "text_embeddings")
        cpp_command = [cpp_sample, convert_model, "Document 1", "Document 2"]
        cpp_result = run_sample(cpp_command)

        # Run JS sample
        js_sample = os.path.join(SAMPLES_JS_DIR, "rag/text_embeddings.js")
        js_command =['node', js_sample, convert_model, "Document 1", "Document 2"]
        js_result = run_sample(js_command)
        
        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        assert py_result.stdout == js_result.stdout, "Python and JS results should match"
