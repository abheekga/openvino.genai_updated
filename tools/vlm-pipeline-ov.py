import logging as log
import sys
import argparse
import os
from transformers import AutoProcessor, LlavaNextProcessor, LlavaNextVideoProcessor
from optimum.intel.openvino import OVWeightQuantizationConfig, OVModelForCausalLM, OVModelForVisualCausalLM
from pathlib import Path
import os

import shutil
import nncf
import openvino as ov
import gc
import glob
from timeit import Timer
from PIL import Image
import numpy as np
import av 
from huggingface_hub import hf_hub_download


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def run_llava_next(ov_model_path, model_id, height, width):

    if model_id == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        model = OVModelForVisualCausalLM.from_pretrained(ov_model_path, device="GPU")
        processor = LlavaNextVideoProcessor.from_pretrained(ov_model_path)
        # define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": "In great detail, describe everything you can see in this video, including color schemes, objects, background scenery, contextual hints, and any emotions it might evoke. Mention spatial arrangement and textures."},
                    {"type": "video"},
                    ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
        video_path = Path("prompts/SampleVideo_1280x720_1mb.mp4")
        container = av.open(video_path)

        # sample uniformly 8 frames from the video, can sample more for longer videos
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
    else:  
        model = OVModelForVisualCausalLM.from_pretrained(ov_model_path, device="GPU")
        processor = LlavaNextProcessor.from_pretrained(ov_model_path)
        # define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": "In great detail, describe everything you can see in this image, including color schemes, objects, background scenery, contextual hints, and any emotions it might evoke. Mention spatial arrangement and textures."},
                    {"type": "image"},
                    ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        image = Image.open(f"images/image_{width}_{height}.jpg")
        inputs_video = processor(text=prompt, images=image, padding=True, return_tensors="pt").to(model.device)


    res_lambda = lambda: model.generate(**inputs_video, min_new_tokens=1, max_new_tokens=1, do_sample=False)
    first_token_latencys = Timer(res_lambda).repeat(repeat=1+3, number=1)[1:]

    res_lambda = lambda: model.generate(**inputs_video, min_new_tokens=129, max_new_tokens=129, do_sample=False)
    rest_token_time = Timer(res_lambda).repeat(repeat=1+3, number=1)[1:]


    print(first_token_latencys)
    print(rest_token_time)
    first_token_latency = sum(first_token_latencys) / len(first_token_latencys)
    rest_token_latency = (sum(rest_token_time) / len(rest_token_time) - first_token_latency) / 128

    output = model.generate(**inputs_video, max_new_tokens=128, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))
    print(f"first_token_latency: {first_token_latency*1000} ms")
    print(f"rest_token_latency: {rest_token_latency*1000} ms")

def export_llava_next_video(ov_model_path):
    MODEL_DIR = Path(ov_model_path)

    if not (MODEL_DIR / "FP16").exists():
        os.system(f"optimum-cli export openvino --model llava-hf/LLaVA-NeXT-Video-7B-hf --weight-format fp16 {MODEL_DIR}/FP16")

    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 64,
        "ratio": 0.6,
    }

    core = ov.Core()
    LANGUAGE_MODEL_PATH_INT4 = MODEL_DIR / "INT4/openvino_language_model.xml"
    LANGUAGE_MODEL_PATH = MODEL_DIR / "FP16/openvino_language_model.xml"
    if not LANGUAGE_MODEL_PATH_INT4.exists():
        ov_model = core.read_model(LANGUAGE_MODEL_PATH)
        ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
        ov.save_model(ov_compressed_model, LANGUAGE_MODEL_PATH_INT4)
        del ov_compressed_model
        del ov_model
        gc.collect()

        copy_model_folder(MODEL_DIR / "FP16", MODEL_DIR / "INT4", ["openvino_language_model.xml", "openvino_language_model.bin"])


def copy_model_folder(src, dst, ignore_file_names=None):
    ignore_file_names = ignore_file_names or []

    for file_name in Path(src).glob("*"):
        if file_name.name in ignore_file_names:
            continue
        shutil.copy(file_name, dst / file_name.relative_to(src))


def export_model_with_optimum(ov_model_path, model_id, weight="int4"):
    if not os.path.exists(ov_model_path):
        os.system(f"optimum-cli export openvino --trust-remote-code --model {model_id} --weight-format {weight} {ov_model_path}")

    return 0

def export_gemma():
    model_id = "google/gemma-3-4b-it"
    quantization_config = OVWeightQuantizationConfig(bits=4, sym=False)

    model = OVModelForVisualCausalLM.from_pretrained(model_id, device="GPU", quantization_config=quantization_config, trust_remote_code=True)
    model.save_pretrained("./models/gemma-3-4b-it")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.save_pretrained("./models/gemma-3-4b-it")

def run_model_with_benchmark(input, output, ov_model_path, prompt_in):
    print(f"Input Size: {input}, Output Size: {output}")

    prompt = f"prompts/{prompt_in}.jsonl"
    
    os.system(f"python benchmark.py -m {ov_model_path} -d GPU -n 3 -ic {output} -pf {prompt}")
    
    return 0

def run_model_with_vlm_benchmark(input, output, ov_model_path, height, width):
    print(f"Input Size: {input}, Output Size: {output}")

    prompt = "In great detail, describe everything you can see in this image, including color schemes, objects, background scenery, contextual hints, and any emotions it might evoke. Mention spatial arrangement and textures."

    image = f"images/image_{width}_{height}.jpg"
    
    os.system(f'python ../../samples/python/visual_language_chat/benchmark_vlm.py -m {ov_model_path} -d GPU -mt {output} -i {image} -p "{prompt}"')
    
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

    if args.model=="gemma3":
        ov_model_path = "models/gemma-3-4b-it"
        model_id = "google/gemma-3-4b-it"
        if not os.path.exists(ov_model_path):
            export_gemma()
        run_model_with_benchmark(args.input, args.output, ov_model_path, "100")
    elif args.model=="minicpm-v":
        ov_model_path = "models/MiniCPM-V-2_6"
        model_id = "openbmb/MiniCPM-V-2_6"
        export_model_with_optimum(ov_model_path, model_id)
        run_model_with_vlm_benchmark(args.input, args.output, ov_model_path, args.height, args.width)
    elif args.model=="llava-llama":
        ov_model_path = "models/llava3-llama-next"
        model_id = "llava-hf/llama3-llava-next-8b-hf"
        export_model_with_optimum(ov_model_path, model_id)
        run_llava_next(ov_model_path, model_id, args.height, args.width)
    elif args.model=="phi3.5-vision":
        ov_model_path = "models/phi3.5-vision"
        model_id = "microsoft/Phi-3.5-vision-instruct"
        export_model_with_optimum(ov_model_path, model_id)
        run_model_with_vlm_benchmark(args.input, args.output, ov_model_path, args.height, args.width)
    elif args.model=="phi4-vision":
        ov_model_path = "models/phi4-multimodal-instruct"
        model_id = "microsoft/Phi-4-multimodal-instruct"
        export_model_with_optimum(ov_model_path, model_id)
        run_model_with_vlm_benchmark(args.input, args.output, ov_model_path, args.height, args.width)
    elif args.model=="llava-video":
        ov_model_path = "models/llava-next-video-7B-ov"
        model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
        if not os.path.exists(ov_model_path):
            export_llava_next_video(ov_model_path)
        ov_model_path = "models/llava-next-video-7B-ov/INT4"
        run_llava_next(ov_model_path, model_id, args.height, args.width)
    else:
        raise(ValueError("Unsupported pipeline"))

    return 0

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--input", default=1024)
    parser.add_argument("--output", default=128)
    parser.add_argument("--height", default=512)
    parser.add_argument("--width", default=512)
    args=parser.parse_args()
    main(args)
