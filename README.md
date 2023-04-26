# faster-whisper-livestream-translator with OBS NVIDIA noise reduction
Translating livestreams with faster-whisper, and dual language subtitles

## This is mostly just a proof of concept

This code is a mess and mostly broken but somebody asked to see a working example of this setup so I dropped it here.

Would love if somebody fixed or re-implemented these main things in any whisper project:

### 1. Noise Reduction. 
Whisper really needs good noise reduction for some live streams or it barely works. VAD is great but doesn't cut it.
### 2. Dual language subtitles.
Preferably by running two full Whisper models in parallel and sychronizing the segments, not like this code.

This is super buggy and constantly disconnects, but I got very distracted by https://github.com/JonathanFly/bark and might forget to ever come back to this.

# Usage

1. Install OBS Studio
2. Install NVIDIA Audio Effects SDK:
https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/
3. Open OBS setup your stream to stream your desktop audio to a custom rtmp server on localhost.
File, Settings, Stream, Service: Custom..., "rtmp://127.0.0.1:12345"
4. Right click 'Desktop Audio' Audio Mixer, select 'Filters'
5. Click + bottom left, add Noise Suppression. Pick NVIDIA Noise removal. I set it to max level 100. ('Start Recording' instead of 'Start Streaming' to check the effect on the audio.)
6. Start python translate_livestream.py 
7. Click Start Streaming

## Dual Subs with segment coloring
![stream_translate_dual](https://user-images.githubusercontent.com/163408/234679630-cf69aaa6-c83f-48b5-865b-499c86b675b3.PNG)
```
python translate_livestream.py --direct_url rtmp://127.0.0.1:12345 --model_size_or_path medium --task transcribe --dual_language_subs --language ko --interval 7 --threshold 0.5 --min_silence_duration_ms 1000 --no_segment_probabilities
```

## Transcription with word probabilities
![stream_translate_transcribe](https://user-images.githubusercontent.com/163408/234680044-d8549991-9676-40a6-904e-f9e6c72884fe.PNG)
```
python translate_livestream.py --direct_url rtmp://127.0.0.1:12345 --model_size_or_path small --task transcribe --interval 8 --threshold 0.5 --min_silence_duration_ms 2000 --word_timestamps True --history_word_size 0
```

## (You can still run without OBS for noise reduction)
```
python translate_livestream.py https://www.twitch.tv/xqc --model_size_or_path medium --task transcribe --interval 7 --threshold 0.5 --min_silence_duration_ms 2000 --word_timestamps True --history_word_size 0
```

# Install
```
git clone https://github.com/JonathanFly/faster-whisper-livestream-translator.git
cd faster-whisper-livestream-translator
pip install -r requirements.txt
```

# How I installed faster-whisper and cuda
```
mamba create --name faster-whisper python=3.10
mamba activate faster-whisper
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c conda-forge cudatoolkit=11.8.0

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8=8.8.1.*-1+cuda11.8

(maybe restart terminal/conda/env)

git clone https://github.com/JonathanFly/faster-whisper-livestream-translator.git
cd faster-whisper-livestream-translator
pip install -r requirements.txt
mamba install ffmpeg
mamba update ffmpeg
```


Based on https://github.com/fortypercnt/stream-translator 
