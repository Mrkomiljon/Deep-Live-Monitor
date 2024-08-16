# Deep-Live-Monitor
If you find this project useful, please give it a star ❤️❤️
## Results  

https://github.com/user-attachments/assets/3b7794fa-c777-45c1-a225-6d4cdd69c85c


https://github.com/user-attachments/assets/02713303-1ff4-4bcc-80c0-d9ffb7eacb1c


https://github.com/user-attachments/assets/69656d9a-2e66-42fd-82c3-259796530960



https://github.com/user-attachments/assets/4b827f33-eb9a-40f3-8606-0bcb9abf1ed6



https://github.com/user-attachments/assets/173891c8-8b39-43d6-8d8a-8ae2594c9379



https://github.com/user-attachments/assets/e50206a1-bf17-440d-8aa9-82f5a9417e8d


## How do I install it?


### I tested it on Windows, RTX 4090, 24Gb. Inference speed is a little bit lower
#### 1.Setup your platform
-   python (3.10 recommended)
-   pip
-   git
-   [ffmpeg](https://www.youtube.com/watch?v=OlNWCpFdVMA) 
-   [visual studio 2022 runtimes (windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
#### 2. Clone Repository
    git clone https://github.com/Mrkomiljon/Deep-Live-Monitor.git
    cd Deep-Live-Monitor

#### 3. Download Models

 1. [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
 2. [inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

Then put those 2 files on the "**models**" folder

#### 4. Install dependency
We highly recommend to work with a  `venv`  to avoid issues.
```
pip install -r requirements.txt
```
##### DONE!!! If you dont have any GPU, You should be able to run roop using `python run.py` command. Keep in mind that while running the program for first time, it will download some models which can take time depending on your network connection.

### *Proceed if you want to use GPU Acceleration
### CUDA Execution Provider (Nvidia)*

1.  Install  [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
    
2.  Install dependencies:
    

```
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3

```

3.  Usage in case the provider is available:

```
python run.py --execution-provider cuda

```

### [](https://github.com/s0md3v/roop/wiki/2.-Acceleration#coreml-execution-provider-apple-silicon)CoreML Execution Provider (Apple Silicon)

1.  Install dependencies:

```
pip uninstall onnxruntime onnxruntime-silicon
pip install onnxruntime-silicon==1.13.1

```

2.  Usage in case the provider is available:

```
python run.py --execution-provider coreml

```

### [](https://github.com/s0md3v/roop/wiki/2.-Acceleration#coreml-execution-provider-apple-legacy)CoreML Execution Provider (Apple Legacy)

1.  Install dependencies:

```
pip uninstall onnxruntime onnxruntime-coreml
pip install onnxruntime-coreml==1.13.1

```

2.  Usage in case the provider is available:

```
python run.py --execution-provider coreml

```

### [](https://github.com/s0md3v/roop/wiki/2.-Acceleration#directml-execution-provider-windows)DirectML Execution Provider (Windows)

1.  Install dependencies:

```
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml==1.15.1

```

2.  Usage in case the provider is available:

```
python run.py --execution-provider directml

```

### [](https://github.com/s0md3v/roop/wiki/2.-Acceleration#openvino-execution-provider-intel)OpenVINO™ Execution Provider (Intel)

1.  Install dependencies:

```
pip uninstall onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino==1.15.0

```

2.  Usage in case the provider is available:

```
python run.py --execution-provider openvino
```

Executing `python run.py` --execution-provider **cuda**  command will launch this window:


- Choose a face image (the face you want to use).
- Choose the target image or video (where you want to replace the face).
- Click Start.
- Open the file explorer and go to the output directory you selected.
- You’ll see a folder named after the video title where the frames are being processed.
- When it’s done, the output file will be ready in that folder.
- That’s it!
## For the monitor mode
You should open any video on desktop


Additional command line arguments are given below. To learn out what they do, check [this guide](https://github.com/s0md3v/roop/wiki/Advanced-Options).

```
options:
  -h, --help                                               show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                     select an source image
  -t TARGET_PATH, --target TARGET_PATH                     select an target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                     select output file or directory
  --frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]  frame processors (choices: face_swapper, face_enhancer, ...)
  --keep-fps                                               keep original fps
  --keep-audio                                             keep original audio
  --keep-frames                                            keep temporary frames
  --many-faces                                             process every face
  --video-encoder {libx264,libx265,libvpx-vp9}             adjust output video encoder
  --video-quality [0-51]                                   adjust output video quality
  --max-memory MAX_MEMORY                                  maximum amount of RAM in GB
  --execution-provider {cpu} [{cpu} ...]                   available execution provider (choices: cpu, ...)
  --execution-threads EXECUTION_THREADS                    number of execution threads
  -v, --version                                            show program's version number and exit
```
