# Benchmarking script for large language models

This script provides a unified approach to estimate performance for Large Language Models.
It is based on pipelines provided by Optimum-Intel and allows to estimate performance for
pytorch and openvino models, using almost the same code and precollected models.

## Usage 

### 1. Start a Python virtual environment
   
``` bash
python3 -m venv python-env
source python-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
> Note:
> If you are using an existing python environment, recommend following command to use all the dependencies with latest versions:  
> pip install -U --upgrade-strategy eager -r requirements.txt

### 2. Convert a model to OpenVINO IR
   
The conversion script for preparing benchmarking models,
`convert.py` allows to reproduce IRs stored on shared drive.

Prerequisites:
install conversion dependencies using `requirements.txt`

Usage:

```bash
python convert.py --model_id <model_id_or_path> --output_dir <out_dir>
```

Paramters:
* `--model_id` - model_id for downloading from huggngface_hub (https://huggingface.co/models) or path with directory where pytorch model located.
* `--output_dir` - output directory for saving OpenVINO model
* `--precision` - (optional, default FP32), precision for model conversion FP32 or FP16
* `--save_orig` - flag for saving original pytorch model, model will be located in `<output_dir>/pytorch` subdirectory.
* `--compress_weights` - The weight compression option, INT8 - INT8 weights, 4BIT_DEFAULT - for 4-bit compression with predefined configs with performance-accuracy trade-off, 4BIT_MAXIMUM - for 4-bit compression with predefined configs for the best performance, INT4_SYM - for INT4 compressed weights with symmetric quantization, INT4_ASYM - for INT4 compressed weights with assymetric quantization. You can specify multiple backends separated by a space.
* `--compress_weights_backends` - (optional, default openvino) backends for weights compression, this option has an effect only with `--compress_weights`. You can specify multiple backends separated by a space.
* `--ratio` - Compression ratio between primary and backup precision, e.g. INT4/INT8.
* `--group_size` - Size of the group of weights that share the same quantization parameters

Usage example:
```bash
python convert.py --model_id meta-llama/Llama-2-7b-chat-hf --output_dir models/llama-2-7b-chat
```

the result of running the command will have the following file structure:

    |-llama-2-7b-chat
      |-pytorch
        |-dldt
           |-FP32
              |-openvino_model.xml
              |-openvino_model.bin
              |-config.json
              |-added_tokens.json
              |-tokenizer_config.json
              |-tokenizer.json
              |-tokenizer.model
              |-special_tokens_map.json

### 3. Benchmarking

Prerequisites:
install benchmarking dependencies using `requirements.txt`

``` bash
pip install -r requirements.txt
```
note: **You can specify the installed OpenVINO version through pip install**
``` bash
# e.g. 
pip install openvino==2023.3.0
```

### 4. Run the following command to test the performance of one LLM model
``` bash
python benchmark.py -m <model> -d <device> -r <report_csv> -f <framework> -p <prompt text> -n <num_iters>
# e.g.
python benchmark.py -m models/llama-2-7b-chat/pytorch/dldt/FP32 -n 2
python benchmark.py -m models/llama-2-7b-chat/pytorch/dldt/FP32 -p "What is openvino?" -n 2
python benchmark.py -m models/llama-2-7b-chat/pytorch/dldt/FP32 -pf prompts/llama-2-7b-chat_l.jsonl -n 2
```
Parameters:
* `-m` - model path
* `-d` - inference device (default=cpu)
* `-r` - report csv
* `-f` - framework (default=ov)
* `-p` - interactive prompt text
* `-pf` - path of JSONL file including interactive prompts
* `-n` - number of benchmarking iterations, if the value greater 0, will exclude the first iteration. (default=0)
* `-ic` - limit the output token size (default 512) of text_gen and code_gen models.


``` bash
python ./benchmark.py -h # for more information
```

## Running `torch.compile()`

The option `--torch_compile_backend` uses `torch.compile()` to speed up
the PyTorch code by compiling it into optimized kernels using a selected backend.

Prerequisites: install benchmarking dependencies using requirements.txt

``` bash
pip install -r requirements.txt
```

In order to run the `torch.compile()` on CUDA GPU, install additionally the nightly PyTorch version:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

Add the option `--torch_compile_backend` with the desired backend: `pytorch` or `openvino` (default) while running the benchmarking script:

```bash
python ./benchmark.py -m models/llama-2-7b-chat/pytorch -d CPU --torch_compile_backend openvino
```

## Run on 2 sockets platform

benchmark.py sets openvino.properties.streams.num(1) by default

| OpenVINO version    | Behaviors                                       |
|:--------------------|:------------------------------------------------|
| Before 2024.0.0 | streams.num(1) <br>execute on 2 sockets. |
| 2024.0.0 | streams.num(1) <br>execute on the same socket as the APP is running on. |

numactl on Linux or --load_config for benchmark.py can be used to change the behaviors.

For example, --load_config config.json as following in OpenVINO 2024.0.0 will result in streams.num(1) and execute on 2 sockets.
```
{"INFERENCE_NUM_THREADS":<NUMBER>}
```
`<NUMBER>` is the number of total physical cores in 2 sockets

## Additional Resources
### 1. NOTE
> If you encounter any errors, please check **[NOTES.md](./doc/NOTES.md)** which provides solutions to the known errors.
### 2. Image generation
> To configure more parameters for image generation models, reference to **[IMAGE_GEN.md](./doc/IMAGE_GEN.md)**
