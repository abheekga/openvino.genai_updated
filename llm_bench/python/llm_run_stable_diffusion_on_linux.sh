#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cat /proc/cpuinfo | grep 'model name' | uniq
grep MemTotal /proc/meminfo

git lfs install
export GIT_LFS_SKIP_SMUDGE=0
echo "Download tiny-sd"
git_clone_stable_diff="git clone https://huggingface.co/segmind/tiny-sd"
echo ${git_clone_stable_diff}
eval ${git_clone_stable_diff}
wait

original_dir="tiny-sd"
dst_dir="./ov_models/tiny-sd"

convert_model="python ./llm_bench/python/convert.py --model_id ${original_dir} --output_dir ${dst_dir} --precision FP16 --stateful"
echo ${convert_model}
eval ${convert_model}
wait

bemchmarking="python ./llm_bench/python/benchmark.py -m ${dst_dir}/pytorch/dldt/FP16/ -pf ./llm_bench/python/prompts/stable-diffusion.jsonl -d cpu -n 1"
echo ${bemchmarking}
eval ${bemchmarking}

rm -rf ${original_dir}
rm -rf ${dst_dir}