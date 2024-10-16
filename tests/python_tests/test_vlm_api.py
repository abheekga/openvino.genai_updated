# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai
import pytest
import gc
import os
import numpy as np
from PIL import Image
from multiprocessing import Process

from openvino_genai import VLMPipeline
from openvino import Tensor
from common import get_greedy, get_image_by_link, get_beam_search, get_greedy, get_multinomial_all_parameters

def get_ov_model(model_dir):
    import sys
    from pathlib import Path
    #TODO: use optimum-intel

    sys.path.append(str(Path(__file__).resolve().parents[2] / 'samples/cpp/visual_language_chat'))
    import importlib
    export_MiniCPM = importlib.import_module("export_MiniCPM-V-2_6", "export_MiniCPM")
    convert_llm = getattr(export_MiniCPM, "convert_llm")
    convert_vision_encoder = getattr(export_MiniCPM, "convert_vision_encoder")
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    import os
    import openvino_tokenizers
    import openvino as ov
    import gc

    model_id = "openbmb/MiniCPM-V-2_6"
    ckpt = Path(os.path.join(model_dir, "ckpt"))
    if not ckpt.exists():
        snapshot_download = getattr(export_MiniCPM, "snapshot_download")
        patch_model_code = getattr(export_MiniCPM, "patch_model_code")
        snapshot_download(model_id, local_dir=ckpt, force_download=True)
        patch_model_code(ckpt)
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
    model.eval()
    model.config.save_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer.save_pretrained(model_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, os.path.join(model_dir, "openvino_tokenizer.xml"))
    ov.save_model(ov_detokenizer, os.path.join(model_dir, "openvino_detokenizer.xml"))
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    processor.save_pretrained(model_dir)

    convert_llm(model, model_dir)
    del model.llm
    gc.collect()

    convert_vision_encoder(model, model_dir)
    return model_dir

sampling_configs = [
    get_beam_search(),
    get_greedy(),
    get_multinomial_all_parameters()
]

prompts = [
    "What is on the image?",
    "What is special about this image?",
    "Tell me more about this image."
]

image_links = [
    "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://github.com/user-attachments/assets/8c9ae017-7837-4abc-ae92-c1054c9ec350"
]

image_links_for_testing = [
    [],
    [image_links[0]],
    [image_links[1], image_links[0]],
    [image_links[0], image_links[2], image_links[1]]
]

@pytest.mark.precommit
def test_vlm_pipeline(tmp_path):
    import os

    def streamer(word: str) -> bool:
        print(word, end="")
        return False

    model_path = get_ov_model(os.path.join(tmp_path, "miniCPM"))

    for generation_config in sampling_configs:
        for links in image_links_for_testing:
            images = []
            for link in links:
                images.append(get_image_by_link(link))

            pipe = VLMPipeline(model_path, "CPU")
            pipe.start_chat()

            pipe.generate(prompts[0], images=images, generation_config=generation_config, streamer=streamer)

            for prompt in prompts[1:]:
                pipe.generate(prompt, generation_config=generation_config, streamer=streamer)

            pipe.finish_chat()
            gc.collect()
    tokenizer = pipe.get_tokenizer()
    tokenizer.encode("")
    del pipe
    gc.collect()


