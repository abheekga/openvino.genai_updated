import logging as log
import sys


    # from optimum.intel.openvino import OVDiffusionPipeline

#from optimum.intel.openvino import OVStableDiffusionXLPipeline
import argparse
import os
import shutil
import glob
import datetime
import psutil
import threading
import time
import subprocess
from pathlib import Path

def measure_memory(output_file_name, stop_event, compile_event):
    with open(output_file_name, mode='w') as output_file:
        while not stop_event.is_set():
            gpu_mem_cmd = r'(((Get-Counter "\GPU Process Memory(*)\Local Usage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'

            def run_command(command):
                val = subprocess.run(['powershell', '-Command', command], capture_output=True).stdout.decode("ascii")
                return float(val.strip().replace(',', '.')) / 2**20

            gpu_memory = run_command(gpu_mem_cmd)
            cpu_memory = (psutil.virtual_memory().total - psutil.virtual_memory().available) / 1024 / 1024
            now = datetime.datetime.now()
            output_file.write(f"Timestamp: {now.time()}\n")
            output_file.write(f"GPU Memory Usage: {gpu_memory:.2f} MB\n")
            output_file.write(f"CPU Memory Usage: {cpu_memory:.2f} MB\n")
            if compile_event.is_set():
                output_file.write("Compilation finished.\n")
                compile_event.clear()
            output_file.flush()  # Ensure data is written immediately

def monitor_compilation_folder(base_folder, compile_event, stop_event):
    """Monitor for compilation_phase subfolder"""
    base_path = Path(base_folder)
    compilation_path = base_path / "compilation_phase"
    
    print(f"Monitoring for compilation phase folder: {compilation_path}")
    
    while not stop_event.is_set() and not compile_event.is_set():
        try:
            if compilation_path.exists() and compilation_path.is_dir():
                print("Compilation_phase folder detected! Stopping memory monitoring...")
                compile_event.set()
                break
        except Exception as e:
            print(f"Error checking compilation folder: {e}")
            
        time.sleep(0.1)  # Check every 100ms

def run_model(input, output, ov_model_path, model_id, weight="int4", task=False, mem=False):
    
    print(f"Input Size: {input}, Output Size: {output}")

    prompt = f"prompts/{input}_tokens_test.jsonl"

    logger = log.getLogger()
    
    if not os.path.exists(ov_model_path):
        if task:
            os.system(f"optimum-cli export openvino --trust-remote-code --model {model_id} --weight-format {weight} --ratio 1.0 --sym --group-size 128 {ov_model_path} --task text-generation-with-past")
        else:
            os.system(f"optimum-cli export openvino --trust-remote-code --model {model_id} --weight-format {weight} --ratio 1.0 --sym --group-size 128 {ov_model_path}")

    if mem:
        folder_path = ov_model_path

        total_size = 0
        for root, dirs, files in os.walk(folder_path):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    total_size += os.path.getsize(file_path)
                except OSError:
                    # Skip files that can't be accessed
                    pass

        size_gb = total_size / (1024 ** 3)
        print(f"Model size: {size_gb:.2f} GB")
        
        monitoring_folder = "memory_logs_temp"
        monitoring_path = Path(monitoring_folder)
        monitoring_path.mkdir(parents=True, exist_ok=True)

        output_file = f"memory_log_output.txt"
        stop_event = threading.Event()
        compile_event = threading.Event()
        logging_thread = threading.Thread(target=measure_memory, args=(output_file, stop_event, compile_event))
        folder_thread = threading.Thread(target=monitor_compilation_folder, args=(monitoring_folder, compile_event, stop_event), daemon=True)
        logging_thread.start()
        folder_thread.start()
        time.sleep(2)  # Ensure logging thread starts before benchmark
        logger.info("Memory logging started.")
        os.system(f"python benchmark.py -m {ov_model_path} -d GPU -n 3 -ic {output} -pf {prompt} -mc 1 -mc_dir memory_logs_temp")
        logger.info("Inference completed.")
        time.sleep(2)  # Give it some time to collect the idle memory just in case
        stop_event.set()
        logging_thread.join()
        folder_thread.join()
        logger.info("Memory logging stopped.")
        # os.system(f"python benchmark_mem.py -m {ov_model_path} -d GPU -n 3 -ic {output} -pf {prompt}")
        if monitoring_path.exists():
            shutil.rmtree(monitoring_path)
        with open(output_file, mode='a') as memory_file:
            memory_file.write(f"Model size: {size_gb:.2f} GB")
    else:
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
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="llama3.2":
        ov_model_path = "models/llama-3.2-3b"
        model_id = "meta-llama/Llama-3.2-3B"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="llama3.1":
        ov_model_path = "models/llama-3.1-8b"
        model_id = "meta-llama/Llama-3.1-8B"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="llama3.0":
        ov_model_path = "models/llama-3.0-8b"
        model_id = "meta-llama/Llama-3-8B"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="glm":
        ov_model_path = "models/glm-edge-4b"
        model_id = "zai-org/glm-edge-4b-chat"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="qwen2.5":
        ov_model_path = "models/Qwen2.5-7B-Instruct"
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="qwen3-0.6":
        ov_model_path = "models/Qwen3-0.6B"
        model_id = "Qwen/Qwen3-0.6B"
        run_model(args.input, args.output, ov_model_path, model_id, weight="fp16", mem=args.mem)
    elif args.model=="qwen3-8":
        ov_model_path = "models/Qwen3-8B"
        model_id = "Qwen/Qwen3-8B"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="phi-3.5":
        ov_model_path = "models/Phi-3.5-mini-instruct"
        model_id = "microsoft/Phi-3.5-mini-instruct"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="phi4-reason":
        ov_model_path = "models/Phi-4-mini-reasoning"
        model_id = "microsoft/Phi-4-mini-reasoning"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="phi4-instruct":
        ov_model_path = "models/Phi-4-mini-instruct"
        model_id = "microsoft/Phi-4-mini-instruct"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="gemma1":
        ov_model_path = "models/gemma1-7b"
        model_id = "google/gemma-7b"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    elif args.model=="mistral":
        ov_model_path = "models/Mistral-7B-Instruct"
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        run_model(args.input, args.output, ov_model_path, model_id, task=True, mem=args.mem)
    elif args.model=="minicpm":
        ov_model_path = "models/Minicpm-1b-sft-bf16"
        model_id = "openbmb/MiniCPM-1B-sft-bf16"
        run_model(args.input, args.output, ov_model_path, model_id, weight="fp16", mem=args.mem)
    elif args.model=="deepseek":
        ov_model_path = "models/Deepseek-R1-Distill-Qwen-14B"
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        run_model(args.input, args.output, ov_model_path, model_id, mem=args.mem)
    else:
        raise(ValueError("Unsupported pipeline"))

    return 0

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--input", default=1024)
    parser.add_argument("--output", default=128)
    parser.add_argument("--mem", default=False, action="store_true")
    args=parser.parse_args()
    main(args)

