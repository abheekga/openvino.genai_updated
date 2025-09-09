import logging as log
import sys


    # from optimum.intel.openvino import OVDiffusionPipeline

#from optimum.intel.openvino import OVStableDiffusionXLPipeline
import argparse
import os
import shutil
import glob


def run_model(input, output, ov_model_path, model_id, weight="int4", task=False):
    
    print(f"Input Size: {input}, Output Size: {output}")

    prompt = f"prompts/{input}_tokens_test.jsonl"

    
    
    if not os.path.exists(ov_model_path):
        if task:
            os.system(f"optimum-cli export openvino --trust-remote-code --model {model_id} --weight-format {weight} {ov_model_path} --task text-generation-with-past")
        else:
            os.system(f"optimum-cli export openvino --trust-remote-code --model {model_id} --weight-format {weight} {ov_model_path}")

    os.system(f"python benchmark.py -m {ov_model_path} -d GPU -n 3 -ic {output} -pf {prompt}")
    return 0

def clear_storage_space():
    # Clears the cache and model directory
    CACHE_DIR = "C:/Users/gta/.cache/huggingface/hub"
    MODEL_DIR = "./models"
    
    TARGET_DIRS = [CACHE_DIR, MODEL_DIR]
    print(f"Clearing {CACHE_DIR} and {MODEL_DIR}")

    for target in TARGET_DIRS:
        abs_path = os.path.abspath(target)
        if os.path.exists(abs_path):
            shutil.rmtree(abs_path, ignore_errors=True)
        else:
            print(f"Directory not found: {abs_path}")

    # Files may be temporarily added to temp app data, clearing only the openvino files
    # from that directory
    TEMP_DIR = os.path.join(os.environ["LOCALAPPDATA"], "Temp")
    
    pattern = os.path.join(TEMP_DIR, "**", "*.bin")

    for file_path in glob.iglob(pattern, recursive=True):
        if "openvino" in file_path.lower():
            try:
                print(f"Deleting {file_path}")
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def main(args):
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    total, used, free = shutil.disk_usage("C:\\")
    threshold = 50 * (1024**3)
    if free < threshold:
        print(f"Free space on C drive is below 50 GB, currently at ({free / (1024**3):.2f}) GB remaining.")
        # User can clear storage space if needed.
        # clear_storage_space()
    else:
        print(f"Free space on C drive is above 50 GB, currently at ({free / (1024**3):.2f}) GB remaining.")


    if args.model=="llama2":
        ov_model_path = "models/llama-2-7b"
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="llama3.2":
        ov_model_path = "models/llama-3.2-3b"
        model_id = "meta-llama/Llama-3.2-3B"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="llama3.1":
        ov_model_path = "models/llama-3.1-8b"
        model_id = "meta-llama/Llama-3.1-8B"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="glm":
        ov_model_path = "models/glm-edge-4b"
        model_id = "zai-org/glm-edge-4b-chat"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="qwen2.5":
        ov_model_path = "models/Qwen2.5-7B-Instruct"
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="qwen3-0.6":
        ov_model_path = "models/Qwen3-0.6B"
        model_id = "Qwen/Qwen3-0.6B"
        run_model(args.input, args.output, ov_model_path, model_id, weight="fp16")
    elif args.model=="qwen3-8":
        ov_model_path = "models/Qwen3-8B"
        model_id = "Qwen/Qwen3-8B"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="phi-3.5":
        ov_model_path = "models/Phi-3.5-mini-instruct"
        model_id = "microsoft/Phi-3.5-mini-instruct"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="phi4-reason":
        ov_model_path = "models/Phi-4-mini-reasoning"
        model_id = "microsoft/Phi-4-mini-reasoning"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="phi4-instruct":
        ov_model_path = "models/Phi-4-mini-instruct"
        model_id = "microsoft/Phi-4-mini-instruct"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="gemma1":
        ov_model_path = "models/gemma1-7b"
        model_id = "google/gemma-7b"
        run_model(args.input, args.output, ov_model_path, model_id)
    elif args.model=="mistral":
        ov_model_path = "models/Mistral-7B-Instruct"
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        run_model(args.input, args.output, ov_model_path, model_id, task=True)
    elif args.model=="minicpm":
        ov_model_path = "models/Minicpm-1b-sft-bf16"
        model_id = "openbmb/MiniCPM-1B-sft-bf16"
        run_model(args.input, args.output, ov_model_path, model_id, weight="fp16")
    elif args.model=="deepseek":
        ov_model_path = "models/Deepseek-R1-Distill-Qwen-14B"
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        run_model(args.input, args.output, ov_model_path, model_id)
    else:
        raise(ValueError("Unsupported pipeline"))

    return 0

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--input", default=1024)
    parser.add_argument("--output", default=128)
    args=parser.parse_args()
    main(args)
