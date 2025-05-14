#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse, json
import openvino_genai
import openvino.properties.hint as hints
import openvino.properties as props
import openvino as ov


def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

def generate_input_data_for_qa(user_content, enable_thinking=False):
    return {
        "messages": [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ],
        "enable_thinking": enable_thinking
    }

def generate_input_data_for_translate(user_content, enable_thinking=False):
    return {
        "messages": [
            {"role": "system", "content": "You are a legal translation assistant. Your task is to translate legal documents between English and Chinese. Ensure the translation is precise, formal, and adheres to legal terminology."},
            {"role": "user", "content": user_content},
        ],
        "enable_thinking": enable_thinking
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(args.model_dir, device, ATTENTION_BACKEND="SDPA")

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 256

    # test for qa
    pipe.start_chat()
    json_input = generate_input_data_for_qa("who you are?", enable_thinking=False)
    prompt = json.dumps(json_input)
    pipe.generate(prompt, config, streamer)
    pipe.finish_chat()
    print('\n')

    # test for translate
    pipe.start_chat()
    json_input = generate_input_data_for_translate("who you are?", enable_thinking=False)
    prompt = json.dumps(json_input)
    pipe.generate(prompt, config, streamer)
    pipe.finish_chat()
    print('\n')


if '__main__' == __name__:
    main()
