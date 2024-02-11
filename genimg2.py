import argparse
import sys
from cuda import cudart
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import pyaudio
import wave
from faster_whisper import WhisperModel
import time
import random
from collections import deque


NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Assuming these imports are from your existing project structure. Adjust as necessary.
from stable_diffusion_pipeline import StableDiffusionPipeline
from utilities import PIPELINE_TYPE, TRT_LOGGER, add_arguments, process_pipeline_args

def initialize_model(prompt):
    args = parseArgs()

    # Process arguments for pipeline initialization
    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = process_pipeline_args(args)

    # Initialize the model
    demo = StableDiffusionXLPipeline(enable_refiner=args.enable_refiner, **kwargs_init_pipeline)

    # Load engines with potential refiner configuration
    kwargs_load_refiner = {'onnx_refiner_dir': args.onnx_refiner_dir, 'engine_refiner_dir': args.engine_refiner_dir} if args.enable_refiner else {}
    demo.loadEngines(
        args.framework_model_dir,
        args.onnx_dir,
        args.engine_dir,
        **kwargs_load_refiner,
        **kwargs_load_engine)

    # Activate engines and load resources
    _, shared_device_memory = cudart.cudaMalloc(demo.get_max_device_memory())
    demo.activateEngines(shared_device_memory)
    demo.loadResources(args.height, args.width, args.batch_size, args.seed)

    return demo


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Txt2Img Demo", conflict_handler='resolve')
    parser = add_arguments(parser)
    parser.add_argument('--version', type=str, default="xl-1.0", choices=["xl-1.0", "xl-turbo"], help="Version of Stable Diffusion XL")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Width of image to generate (must be multiple of 8)")
    parser.add_argument('--num-warmup-runs', type=int, default=1, help="Number of warmup runs before benchmarking performance")
    parser.add_argument('--guidance-scale', type=float, default=5.0, help="Value of classifier-free guidance scale (must be greater than 1)")
    parser.add_argument('--enable-refiner', action='store_true', help="Enable SDXL-Refiner model")
    parser.add_argument('--image-strength', type=float, default=0.3, help="Strength of transformation applied to input_image (must be between 0 and 1)")
    parser.add_argument('--onnx-refiner-dir', default='onnx_xl_refiner', help="Directory for SDXL-Refiner ONNX models")
    parser.add_argument('--engine-refiner-dir', default='engine_xl_refiner', help="Directory for SDXL-Refiner TensorRT engines")
    return parser.parse_args()

class StableDiffusionXLPipeline(StableDiffusionPipeline):
    def __init__(self, vae_scaling_factor=0.13025, enable_refiner=False, **kwargs):
        super().__init__()  # Adjust according to your superclass's __init__ method
        self.enable_refiner = enable_refiner
        self.nvtx_profile = kwargs.get('nvtx_profile', False)
        self.base = StableDiffusionPipeline(
            pipeline_type=PIPELINE_TYPE.XL_BASE,
            vae_scaling_factor=vae_scaling_factor,
            return_latents=self.enable_refiner,
            **kwargs)
        if self.enable_refiner:
            self.refiner = StableDiffusionPipeline(
                pipeline_type=PIPELINE_TYPE.XL_REFINER,
                vae_scaling_factor=vae_scaling_factor,
                return_latents=False,
                **kwargs)

    def loadEngines(self, framework_model_dir, onnx_dir, engine_dir, onnx_refiner_dir='onnx_xl_refiner', engine_refiner_dir='engine_xl_refiner', **kwargs):
        self.base.loadEngines(engine_dir, framework_model_dir, onnx_dir, **kwargs)
        if self.enable_refiner:
            self.refiner.loadEngines(engine_refiner_dir, framework_model_dir, onnx_refiner_dir, **kwargs)

    def activateEngines(self, shared_device_memory=None):
        self.base.activateEngines(shared_device_memory)
        if self.enable_refiner:
            self.refiner.activateEngines(shared_device_memory)

    def loadResources(self, image_height, image_width, batch_size, seed):
        self.base.loadResources(image_height, image_width, batch_size, seed)
        if self.enable_refiner:
            self.refiner.loadResources(image_height, image_width, batch_size, seed + 1 if seed is not None else None)

    def get_max_device_memory(self):
        max_device_memory = self.base.calculateMaxDeviceMemory()
        if self.enable_refiner:
            max_device_memory = max(max_device_memory, self.refiner.calculateMaxDeviceMemory())
        return max_device_memory

    def run(self, prompt, negative_prompt, height, width, batch_size, batch_count, num_warmup_runs, use_cuda_graph, **kwargs_infer_refiner):
        if not isinstance(prompt, list):
            raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
        prompt = prompt * batch_size

        if not isinstance(negative_prompt, list):
            raise ValueError(f"`--negative-prompt` must be of type `str` list, but is {type(negative_prompt)}")
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt * batch_size

        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                images, _ = self.base.infer(prompt, negative_prompt, height, width, warmup=True)
                if self.enable_refiner:
                    images, _ = self.refiner.infer(prompt, negative_prompt, height, width, input_image=images, warmup=True, **kwargs_infer_refiner)

        ret = []
        for _ in range(batch_count):
            print("[I] Running StableDiffusionXL pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            latents, time_base = self.base.infer(prompt, negative_prompt, height, width, warmup=False)
            if self.enable_refiner:
                images, time_refiner = self.refiner.infer(prompt, negative_prompt, height, width, input_image=latents, warmup=False, **kwargs_infer_refiner)
                ret.append(images)
            else:
                ret.append(latents)

            if self.nvtx_profile:
                cudart.cudaProfilerStop()
            if self.enable_refiner:
                print('|-----------------|--------------|')
                print('| {:^15} | {:>9.2f} ms |'.format('e2e', time_base + time_refiner))
                print('|-----------------|--------------|')
        return ret

    def teardown(self):
        self.base.teardown()
        if self.enable_refiner:
            self.refiner.teardown()

def generate_image_with_existing_model(demo, prompt, version="xl-turbo", onnx_dir="onnx-sdxl-turbo", engine_dir="engine-sdxl-turbo", denoising_steps=1, scheduler="EulerA", guidance_scale=0.0, width=1024, height=1024):
    args = parseArgs()  # Ensure this correctly handles or simulates command-line arguments.

    # Ensure prompt is a list
    if not isinstance(prompt, list):
        prompt = [prompt]  # Wrap the prompt in a list

    # Generate a corresponding negative_prompt list with empty strings
    negative_prompt = [''] * len(prompt)

    # Assuming args_run_demo can be derived without needing to reinitialize everything
    args_run_demo = [prompt, negative_prompt, args.height, args.width, args.batch_size, 1, args.num_warmup_runs, False]  # Example, adjust as needed

    kwargs_infer_refiner = {'image_strength': args.image_strength} if args.enable_refiner else {}
    images = demo.run(*args_run_demo, **kwargs_infer_refiner)

    return images

def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=5)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def record_chunk(p, stream):
    print("Recording... Press Ctrl+C to stop.")
    frames = []
    try:
        while True:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
    except KeyboardInterrupt:
        print("Recording stopped.")
    
    # Define the file path internally
    file_path = "temp_chunk.wav"
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()
    return file_path  # Now the function returns the path of the recorded file

# Initialize a deque with a maximum length of 60 characters
transcription_buffer = deque(maxlen=100)

if __name__ == "__main__":
    # Initialize the model once
    demo = initialize_model("anime")
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    try:
        while True:  # Start an infinite loop
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream)
            transcription = transcribe_chunk(model, chunk_file)
            os.remove(chunk_file)
                
                    # Update the transcription buffer
            for char in transcription:
                transcription_buffer.append(char)
                    # Convert the buffer back to a string
            transcription_window = ''.join(transcription_buffer)
            print(NEON_GREEN + transcription_window + RESET_COLOR) 
        # Proceed with image generation using the provided prompt
            images = generate_image_with_existing_model(demo, transcription_window + ",")
            print("Image generation completed.")
    except KeyboardInterrupt:
            print("Stopping...")
    finally:
            stream.stop_stream()
            stream.close()
            p.terminate()