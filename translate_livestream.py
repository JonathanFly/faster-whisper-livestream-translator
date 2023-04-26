import ast
from tabulate import tabulate
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
import argparse
import sys
import signal
from datetime import datetime
import time
import ffmpeg
import numpy as np
from faster_whisper import WhisperModel

import streamlink
import subprocess
import threading

from collections import deque
from numpy_ringbuffer import RingBuffer

import logging
import traceback
import os

import click

from rich import print
from rich.console import Console
import colorsys
console = Console()
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# This is a mess of a fork of https://github.com/fortypercnt/stream-translator
# TODO
# 
# "Stream Ended" messages out of the blue
# Removing the internal VAD was a bad idea. It's a better way to create our intervals than the --interval command. Add it back, with a larger default setting than faster-whisper.
# In retrospect I regret using Click
# faster-whisper is twice as fast if you run the processes in parallel, rather htan like this.

# last minute
# show transcribe + translate command line
# screenshots
# basic colors with config options
# quick test on new system


whisper_model_param = "__WHISPER MODEL:"
whisper_transcribe_param = "__WHISPER TRANSCRIBE: "
silero_param = "__SILERO VAD: "


# Audio Functions
def has_repetition(buffer):
    return len(set(buffer)) != len(buffer)


def ffmpeg_input(input_stream, ffmpeg_loglevel, **kwargs):
    return (
        ffmpeg.input(input_stream, **kwargs)
        .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
        .global_args('-nostdin')  # Disable interaction on standard input
        .global_args('-loglevel', ffmpeg_loglevel)  # Set log level to 'info'

    )

def writer(streamlink_proc, ffmpeg_proc, ffmpeg_loglevel):
    if ffmpeg_proc is None:
        raise ValueError("ffmpeg_proc is not initialized properly")

    while (not streamlink_proc.poll()) and (not ffmpeg_proc.poll()):
        try:
            chunk = streamlink_proc.stdout.read(1024)


            if ffmpeg_loglevel == 'trace':
                logging.debug("Read 1024 bytes from streamlink process.")
            ffmpeg_proc.stdin.write(chunk)


            if ffmpeg_loglevel == 'trace':
                logging.debug("Read 1024 bytes from streamlink process.")

        except (BrokenPipeError, OSError) as e:
            logging.warning(f"Warning: {e}")

        

def create_ffmpeg_process(input_stream, ffmpeg_loglevel, **kwargs):
    try:
        process = ffmpeg_input(input_stream, ffmpeg_loglevel,
                               **kwargs).run_async(pipe_stdout=True, pipe_stdin=True)
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return process


def open_stream(stream, direct_url, preferred_quality, ffmpeg_loglevel, **kwargs):
    if direct_url:
        process = create_ffmpeg_process(
            stream, ffmpeg_loglevel, re=None, listen=1, **kwargs)
        return process, None

    stream_options = streamlink.streams(stream)
    if not stream_options:
        logging.error("No playable streams found on this URL:", stream)
        sys.exit(0)

    option = next((quality for quality in [preferred_quality, 'audio_only', 'audio_mp4a', 'audio_opus', 'best']
                   if quality in stream_options), None)
    if option is None:
        # Fallback
        option = next(iter(stream_options.values()))

    cmd = ['streamlink', stream, option, "-O"]
    streamlink_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    ffmpeg_process = create_ffmpeg_process("pipe:", ffmpeg_loglevel)

    return ffmpeg_process, streamlink_process


def handle_termination(ffmpeg_process, streamlink_process=None):
    def handler(signum, frame):
        ffmpeg_process.kill()
        if streamlink_process:
            streamlink_process.kill()
        sys.exit(0)

    return handler

# Transcription and Translation Functions




def get_color(avg_logprob):
    hue = (avg_logprob + 1) / 2  # Normalize the log probability to the range [0, 1]
    rgb = colorsys.hsv_to_rgb(hue, 1, 1)  
    return "rgb({:.0f},{:.0f},{:.0f})".format(*[x * 255 for x in rgb])  

def get_color_word (prob):
    hue = 0.333 * (prob) 
    rgb = colorsys.hsv_to_rgb(hue, 1, 1) 
    return "rgb({:.0f},{:.0f},{:.0f})".format(*[x * 255 for x in rgb])  


def process_and_print_transcription(model, in_bytes, audio_buffer, previous_text, audio, decode_options, output_text=None, dual_language_subs=False, no_segment_probabilities=False, no_color = False, word_by_word = False, history_word_size = 10, whisper_lag = 0, interval = 5):
    def update_previous_text(previous_text, decoded_text, max_words=13):
        if max_words == 0: return ''
        if previous_text is None:
            previous_text = ''
        combined_words = (previous_text + ' ' + decoded_text).split()
        return ' '.join(combined_words[-max_words:])

    clear_buffers = False

    start_time = datetime.utcnow()

    decoded_language = ""
    # if audio_info.language:
    #    decoded_language = "(" + audio_info.language + ")"

    


    tasks = []
    if dual_language_subs: tasks = ["transcribe", "translate"]
    else: tasks = [decode_options["task"]]

    low_confidence = False
    firstTime = True
    decoded_text = ""
    dualsub_text = ""
    for task in tasks:
        logger.debug(tasks)
        logger.debug(f"Decoding {task}...")
        logger.debug(decode_options)
        decode_options["task"] = task
        if previous_text.strip != "": 
            logging.debug(f"Previous Text: {previous_text}")
            logging.debug(f"history_word_size: {history_word_size}")
            previous_text = update_previous_text(previous_text, "", max_words=history_word_size) 
            logging.debug(f"new Text: {previous_text}")
            decode_options["prefix"] = previous_text
        else: 
            decode_options["prefix"] = ""
        if dual_language_subs and task == "translate":  # TODO Track prev for each task seperately
            decode_options["prefix"] = ""
        segments, audio_info = model.transcribe(audio, **decode_options)
        last_color = 15 #really just helps keep track of the dual sub translation, so you can better match longer captions.
        for segment in segments:
            #console.print(segment)


            if "word_timestamps" in decode_options.keys() and decode_options["word_timestamps"] is True:
                words = segment.words
                for word in words:
                    text = word.word
                    word_prob = word.probability

                    if no_color:
                        console.print(text, end='')
                    elif no_segment_probabilities:
                        console.print(text, end='', style=f"color({last_color})")
                        last_color -= 1
                        if last_color < 1: last_color = 15
                        
                    else: 
                        color = get_color_word(word_prob)
                        console.print(text, end='', style=color)
                    #console.print(segment.avg_logprob)
                    #for word in segment.words:
                    #    print(word.word, end='')
                if firstTime: decoded_text += text
                else: dualsub_text += text


            else:
                
                text = segment.text
                avg_logprob = segment.avg_logprob
                if avg_logprob >  decode_options["log_prob_threshold"]:
                    if no_color:
                        console.print(text, end='')
                    elif no_segment_probabilities:
                        console.print(text, end='', style=f"color({last_color})")
                        last_color -= 1
                        if last_color < 1: last_color = 15
                        
                    else: 
                        color = get_color(avg_logprob)
                        console.print(text, end='', style=color)
                    #console.print(segment.avg_logprob)
                    #for word in segment.words:
                    #    print(word.word, end='')

                else:
                    logger.debug(f"Skipping due to low logprob: {avg_logprob}")
                    low_confidence = True # quick hack to reset history early
                if firstTime: decoded_text += text
                else: dualsub_text += text
                ###
        decoded_text = decoded_text.strip()
        dualsub_text = dualsub_text.strip()
        missing_sub_pair = ''
        if decoded_text != "" or dualsub_text != "":
            lag_text = ''

            if whisper_lag > (interval * 2):
                lag_text = f"  (-{round(whisper_lag)}s)"

            if (dualsub_text == "" and not firstTime) or (decoded_text == "" and firstTime):
                missing_sub_pair = "\n..."



            console.print(f"{missing_sub_pair}{lag_text}")
            if not firstTime: console.print(f"") # print extra new line to space out out dual subtitle pairs for readability

        if firstTime: 
            firstTime = False


    logging.debug(f"\ndecoded_text: {decoded_text}")
    logging.debug(f"\nprevious_text: {previous_text}")
    if decoded_text != "" and previous_text.strip() != decoded_text and low_confidence != True:


        
        previous_text = update_previous_text(previous_text, decoded_text, max_words=history_word_size)
        logging.debug(f"\nNew previous_text: {previous_text}")

    else:
        logger.debug("resetting previous_text")
        previous_text = ''





    #previous_text.append(new_prefix)
    end_time = datetime.utcnow()

    interval = end_time - start_time
    interval_seconds = interval.total_seconds()

    num_channels = 1
    bytes_per_sample = 2
    samples = len(in_bytes) // (bytes_per_sample * num_channels)
    audio_length_in_seconds = samples / SAMPLE_RATE

    # print(f"Audio File Length: {audio_length_in_seconds}")
    # print(f"Interval In Seconds: {interval_seconds}")
    # clear_buffers is never set right now

    #if clear_buffers or has_repetition(previous_text):
    #    print('.', end='')
    #    audio_buffer.reset()
    #    previous_text.reset()
    return previous_text

def whisper_in_seconds(whisper_bytes):
    num_channels = 1
    bytes_per_sample = 2
    samples = whisper_bytes // (bytes_per_sample * num_channels)
    samples_in_seconds = samples / SAMPLE_RATE
    return samples_in_seconds

def main(url, our_loglevel='WARNING', whisper_loglevel='WARNING', ffmpeg_loglevel='WARNING', interval=5, history_word_size = 10, history_buffer_size=0, preferred_quality="audio_only",
         direct_url=False, output_text=None, dual_language_subs=False, no_segment_probabilities=False, no_color=False, lag_skip_ahead = 0, **decode_options):

    logging.basicConfig(level=our_loglevel,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    from click.core import ParameterSource
    ctx = click.get_current_context()

    main_params = locals().copy()
    main_params.pop('decode_options', None)
    log_dict_values(main_params, "raw main() params at start", "Value")
    log_dict_values(
        decode_options, "raw decode_options at start of main", "Value")

    n_bytes = interval * SAMPLE_RATE * 2

    audio_buffer = RingBuffer(
        (history_buffer_size // interval) + 1, dtype=object)
    #previous_text = RingBuffer(history_buffer_size // interval, dtype=object)

    #previous_text = RingBuffer(history_buffer_size // interval, dtype=object)

    # find tagged WhisperModelParam options, build vad_parameters, set it in the main decode_options
    whisper_model_params = {}
    whisper_model_params_to_remove = []
    for option in ctx.command.params:
        if isinstance(option, whisperModelParam):

            param_source = ctx.get_parameter_source(option.name)
            if param_source == ParameterSource.DEFAULT and option.name != "model_size_or_path":
                whisper_model_params_to_remove.append(option.name)
            else:
                whisper_model_params[option.name] = ctx.params[option.name]

    logging.warning(f"Loading model")
    log_dict_values(whisper_model_params,
                    "User Specified WhisperModel Load Param", "Value")

  

    from faster_whisper import WhisperModel
    model = WhisperModel(**whisper_model_params)

    # Whisper logs way to much on INFO for this use case
    logging.getLogger('faster_whisper').setLevel(whisper_loglevel)

    # logging.debug(f"interval: {interval} | history_buffer_size: {history_buffer_size}")

    # find whisper taggged value, delete from out decode_options entirely if they are default value, or tagged silero
    whisper_transcribe_params_to_remove = []
    for option in ctx.command.params:
        if (isinstance(option, whisperParam) or isinstance(option, sileroParam) or isinstance(option, whisperModelParam)):
            param_source = ctx.get_parameter_source(option.name)
            if (option.name in ["vad_filter", "beam_size", "log_prob_threshold"]):
                continue
            if (param_source == ParameterSource.DEFAULT or isinstance(option, sileroParam) or isinstance(option, whisperModelParam)):
                whisper_transcribe_params_to_remove.append(option.name)

    # user Whisper overrides
    user_specified_decode_options = decode_options.copy()
    user_specified_decode_options = {k: v for k, v in decode_options.items(
    ) if k not in whisper_transcribe_params_to_remove}

    # find tagged silero options, build vad_parameters, set it in the main decode_options
    vad_parameters = {}
    silero_params_to_remove = []
    for option in ctx.command.params:
        if isinstance(option, sileroParam):

            param_source = ctx.get_parameter_source(option.name)
            if param_source == ParameterSource.DEFAULT:
                silero_params_to_remove.append(option.name)
            else:
                vad_parameters[option.name] = ctx.params[option.name]

    user_specified_decode_options['vad_parameters'] = vad_parameters

    # this way too much logging but i was getting so confused what parameters were actually being passed to faster whisper
    # Omits all other options, leave defaults to faster whisper
    log_dict_values(user_specified_decode_options,
                    "User Specified Whisper Param", "Value")

    print(whisper_model_params)
    print(user_specified_decode_options)

    # just to log THESE ARE NOT BEING SET BY ANYBODY. LEAVING ALL NON SPECIFIC PARAMETERS TO FASTER-WHISPER DEFAULTS
    whisper_passthrough_default_params = {}
    for param in whisper_transcribe_params_to_remove:
        whisper_passthrough_default_params[param] = ""

    for param in silero_params_to_remove:
        whisper_passthrough_default_params['vad_parameters'] = ""

    log_dict_values(whisper_passthrough_default_params,
                    "Unspecified Whisper Param", "(Whisper Default)")
    # log_dict_values(decode_options,"Original Decode Values", "User + Our Default Values")

    previous_text = ""
    while True:

        console.print(f"Opening stream {url}")
        if 'rtmp' in url:
            console.print(f"[bold red]--> Start (or restart) your OBS local stream now.")

        ffmpeg_process, streamlink_process = open_stream(
            url, direct_url, preferred_quality, ffmpeg_loglevel)

        if streamlink_process:
            writer_thread = threading.Thread(target=writer, args=(
                streamlink_process, ffmpeg_process, ffmpeg_loglevel))
            writer_thread.start()




        signal.signal(signal.SIGINT, handle_termination(
            ffmpeg_process, streamlink_process))
        start_time = datetime.utcnow()
        try:
            first_audio_received = False

            processed_whisper_bytes = 0
            while ffmpeg_process.poll() is None:
                in_bytes = ffmpeg_process.stdout.read(n_bytes)
                if not in_bytes:
                    break

                if not first_audio_received:
                    console.rule("[bold red]Received first audio data",style="bold green")

                    first_audio_received = True
                    start_time = datetime.utcnow()


                audio = np.frombuffer(in_bytes, np.int16).flatten().astype(
                    np.float32) / 32768.0

 
                cur_time = datetime.utcnow()
                process_time = cur_time - start_time
                interval_seconds = process_time.total_seconds()
                whisper_lag = interval_seconds - whisper_in_seconds(processed_whisper_bytes)
                #console.print(f"{round(whisper_lag)}s...")
                if (whisper_lag > (interval * 3)):
                    # prefix is bugged I think
                    previous_text = ''
                    decode_options["prefix"] = ""
                    console.print(f"{round(whisper_lag)}s...")

                skip_ahead = False
                if (lag_skip_ahead > 0 and whisper_lag > lag_skip_ahead):
                    previous_text = ''
                    decode_options["prefix"] = ""
                    console.print(f"Skipping to catch up {round(whisper_lag)}s")
                    skip_ahead = True

                try:
                    if skip_ahead: 
                        previous_text = ''
                    else: 
                        previous_text = process_and_print_transcription(
                            model, in_bytes, audio_buffer, previous_text, audio, user_specified_decode_options, output_text=output_text, dual_language_subs=dual_language_subs, no_segment_probabilities = no_segment_probabilities, no_color = no_color, history_word_size = history_word_size, whisper_lag = whisper_lag, interval = interval)
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logging.warning(f"Transcription Error: {e}")

                    logging.debug(f"Transcription Error: {e}\nTraceback:\n{tb_str}")
                
                processed_whisper_bytes += len(in_bytes)

            logging.warning("Stream ended. cleaning up...")

        except Exception as e:
            tb_str = traceback.format_exc()
            logging.warning(f"Stream Error: {e}")

            logging.debug(f"Stream Error: {e}\nTraceback:\n{tb_str}")
        finally:
            handle_termination(ffmpeg_process, streamlink_process)(None, None)


def parse_temperature(ctx, param, value):
    try:
        if isinstance(value, str):
            temp_values = list(map(float, value.split(',')))
            return temp_values
        else:
            return value
    except ValueError:
        raise click.BadParameter(
            'Temperature must be a float or a list of floats separated by commas.')


def log_dict_values(dictionary, left_desc, right_desc):
    max_line_length = 60
    spacer = '-' * max_line_length

    formatted_data = [[k, v] for k, v in dictionary.items()]
    table = tabulate(formatted_data, tablefmt="simple",
                     headers=[f"{left_desc}", f"{right_desc}"])
    logger.debug(f"\n{spacer}\n{table}\n{spacer}\n")


def parse_suppress_tokens(ctx, param, value):
    try:
        suppress_tokens = ast.literal_eval(value)
        if not isinstance(suppress_tokens, list):
            raise ValueError("suppress_tokens must be a list of integers")
        for token in suppress_tokens:
            if not isinstance(token, int):
                raise ValueError(
                    "All elements in suppress_tokens must be integers")
        return suppress_tokens
    except (ValueError, SyntaxError) as e:
        raise click.BadParameter(
            f"Invalid value for --suppress_tokens: {value}. Error: {e}")


class whisperModelParam(click.Option):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class whisperParam(click.Option):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class sileroParam(click.Option):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@click.command(
    help="stream-translator with OBS\n\n"
    "NVIDIA Noise Reduction. \n\n"

    "This is a mess and super buggy, kind of slow. But somebody asked to post an example so here it is.\n\n"

    "Especially the audio and stream handling is pretty broken, you have to reconnet or restart the OBS stream a lot. \n\n Don't assume any setting or parameter is best, or is working, it's probably doing nothing.\n\n"
    "I was testing different overlapping windows and different chunking strategies but yanked all that.\n\n"
    "(So history_buffer_size and numpy_ringbuffer isn't even being used.)\n\n"
    "I only just checked that I had left this in a working state.\n\n"
    "I was planning on coming back to this but I got very distracted by the new bark model https://github.com/JonathanFly/bark/tree/main/bark \n\n"
    "I realized I might never come back to this and a half-broken example is better than nothing.\n\n"
    "I do occasionally fire this up so this should mostly work.\n\n"
    "I ran this on a 3090, haven't really tested what happenes when your GPU can't keep up.\n\n"
    "\n\n",

    epilog=(
        "Useful Parameters to Tweak:\n\n"
        "--dual_language_subs:\n\n"
        "   Use to get both languages in the subtitles, like the Language Reactor Netflix Plugin.\n\n"
        "--interval:\n\n"
        "   Whisper transcribes every X seconds. Default 5.\n\n"
        "   Lower values reduce latency but reduce quality and increase GPU usage\n\n"
        "--beam_size:\n\n"
        "   Default 5. You can set it to 1 to run a lot faster, quality is only a little worse.\n\n"
        "--min_silence_duration_ms:\n\n"
        "   Default, wait for 2 seconds of no speech before breaking up the audio into chunk \n\n"
        "--threshold\n\n"
        "   Probabilities ABOVE this value are considered as SPEECH. 0.5 is pretty good for most datasets\n\n"
        "For OBS Noise Reduction, Start translator_obs first, then click 'Start Streaming' in OBS to the same address and port\n\n"
        "Something like:\n\n"
        "python translate_livestream.py --direct_url rtmp://127.0.0.1:12345 --model_size_or_path medium --task transcribe --dual_language_subs --language ko --interval 7 --threshold 0.5 --min_silence_duration_ms 1000 --no_segment_probabilities\n\n"

        "python translate_livestream.py --direct_url rtmp://127.0.0.1:12345 --model_size_or_path small --task transcribe --interval 8 --threshold 0.5 --min_silence_duration_ms 2000 --word_timestamps True --history_word_size 0\n\n"

        "It's twice as fast if you run two instances of whisper, one for each language, instead of using the dual sub parameter."
        "\n\n"
    )
)
# translator params
@click.argument('url')
@click.option('--direct_url', is_flag=True, help='Pass URL directly to ffmpeg. Otherwise, streamlink.')
@click.option('--preferred_quality', show_default=True, default='audio_only', help='Stream quality. "best"/"worst" always available. Type "streamlink URL" for options.')
@click.option('--dual_language_subs', is_flag=True, help='Show both the transcription and the translation. (Twice as slow, but workable with faster-whisper.)')
@click.option('--no_segment_probabilities', is_flag=True, help='Just go through a cycle of colors instead of showing probabilities.')
@click.option('--no_color', is_flag=True, help='no color text')


@click.option('--interval', type=int, show_default=True, default=6, help='Audio Interval between calls (seconds).')
@click.option('--history_word_size', type=int, show_default=True, default=0,
              help='Keep the last N words as history. Set to lower or 0 if you get a lot of repetition. Not sure if this helps translation quality or just transcription. Default 0 because buggy.')


@click.option('--history_buffer_size', type=int, show_default=True, default=0,
              help='(Not Implemented) Previous audio/text in seconds for model conditioning. May cause repetition/loops.')
@click.option('--lag_skip_ahead', type=int, show_default=True, default=0,
              help="If we think we're lagging behind by this much, skip next next segment to try and catch up. Probably buggy.")
@click.option('--output_text', show_default=True, default=None, type=click.Path(), help='Write captions to a text file.')
@click.option('--our_loglevel', show_default=True, default='WARNING', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help="Try INFO or DEBUG if you don't why things keep breaking.")
@click.option('--whisper_loglevel', show_default=True, default='WARNING', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help="Try INFO or DEBUG if you don't why things keep breaking. ")
@click.option('--ffmpeg_loglevel', show_default=True, default='panic', type=click.Choice(['quiet', 'panic', 'fatal', 'error', 'warning', 'info', 'verbose', 'debug', 'trace']), help="ffmpeg loglevel")
# Load WhisperModel faster-whisper params
@click.option('--model_size_or_path', cls=whisperModelParam, show_default=True, default='medium', type=click.Choice(['tiny', 'tiny.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2']), help=f'{whisper_model_param}Transcription model. You can also specify a faster whisper model path.')
@click.option('--device',  cls=whisperModelParam, show_default=True, default='cuda', type=click.Choice(['cuda', 'cpu', 'auto']),
              help=f'{whisper_model_param}Faster-whisper device')
@click.option('--device_index',  cls=whisperModelParam, type=int, show_default=True, default=0, help=f'{whisper_model_param}Device ID to use')
@click.option('--compute_type',  cls=whisperModelParam, show_default=True, default='float16', type=click.Choice(['int8', 'int8_float16', 'int16', 'float16']),
              help=f'{whisper_model_param}Quantization type.')
@click.option('--cpu_threads',  cls=whisperModelParam, type=int, show_default=True, default=4, help=f'{whisper_model_param}Number of threads to use when running on CPU.')
@click.option('--num_workers',  cls=whisperModelParam, type=int, show_default=True, default=1, help=f'{whisper_model_param}for multiple Python threads')
@click.option('--download_root',  cls=whisperModelParam, show_default=True, default=None, type=click.Path(), help=f'{whisper_model_param}Write captions to a text file.')
# faster-whisper model.transcribe() params, all passed through as is.
@click.option('--language', cls=whisperParam,  show_default=True, default=None, help=f"{whisper_transcribe_param}Language code like 'en' or 'fr', default autodetect.")
@click.option('--task', show_default=True, default='translate', cls=whisperParam,  type=click.Choice(['transcribe', 'translate']),
              help=f'{whisper_transcribe_param}Transcribe (original) or translate to English.')
@click.option('--beam_size', cls=whisperParam, type=int, show_default=True, default=1, help=f'{whisper_transcribe_param}Beam search size. 1: use greedy algorithm, pretty fast and nearly as good.')
@click.option('--best_of', cls=whisperParam, type=int, show_default=True, default=5, help=f'{whisper_transcribe_param}Candidates for sampling with non-zero temperature.')
@click.option('--patience', cls=whisperParam, type=float, show_default=True, default=1, help=f'{whisper_transcribe_param}beam search patience factor')
@click.option('--length_penalty', cls=whisperParam, type=float, show_default=True, default=1, help=f'{whisper_transcribe_param}length factor')
@click.option('--temperature', cls=whisperParam, type=str, show_default=True, default='0.0,0.2,0.4,0.6,0.8,1.0', callback=parse_temperature, help='{whisper_transcribe_param} Temperature for sampling')
@click.option('--compression_ratio_threshold', cls=whisperParam, type=float, show_default=True, default=2.4, help=f'{whisper_transcribe_param}Threshold for gzip compression ratio to treat as failed.')
@click.option('--log_prob_threshold', cls=whisperParam, type=float, show_default=True, default=-0.92, help=f'{whisper_transcribe_param}Threshold for average log probability over sampled tokens to treat as failed.')
@click.option('--no_speech_threshold', cls=whisperParam, type=float, show_default=True, default=0.6, help=f'{whisper_transcribe_param}Threshold for no_speech probability to consider the segment as silent.')
@click.option('--condition_on_previous_text', cls=whisperParam, type=bool, show_default=True, default=True, help=f'{whisper_transcribe_param}If True, provide previous output as a prompt for the next window.')
@click.option('--initial_prompt', cls=whisperParam, type=str, show_default=True, default=None, help='   WHISPER: Optional text to provide as a prompt for the first window.')
@click.option('--prefix', cls=whisperParam, type=str, show_default=True, default=None, help=f'{whisper_transcribe_param}Optional text to provide as a prefix for the first window.')
@click.option('--suppress_blank', cls=whisperParam, type=bool, show_default=True, default=True, help=f'{whisper_transcribe_param}Suppress blank outputs at the beginning of the sampling.')
@click.option('--suppress_tokens', cls=whisperParam, type=str, show_default=True, default=[-1], callback=parse_suppress_tokens, help=f'{whisper_transcribe_param}List of token IDs to suppress. -1 suppresses a default set.')
@click.option('--without_timestamps', cls=whisperParam, type=bool, show_default=True, default=False, help=f'{whisper_transcribe_param}Only sample text tokens.')
@click.option('--max_initial_timestamp', cls=whisperParam, type=float, show_default=True, default=1.0, help=f'{whisper_transcribe_param}Initial timestamp cannot be later than this value.')
@click.option('--word_timestamps', cls=whisperParam, type=bool, show_default=True, default=False, help=f'{whisper_transcribe_param}Extract word-level timestamps and include them in each segment.')
@click.option('--prepend_punctuations', cls=whisperParam, type=str, show_default=True, default="\"'.。,，!！?？:：”)]}、", help=f'{whisper_transcribe_param}Merge these punctuation symbols with the next word.')
@click.option('--append_punctuations', cls=whisperParam, type=str, show_default=True, default="\"'.。,，!！?？:：”)]}、", help=f'{whisper_transcribe_param}Merge these punctuation symbols with the previous word.')
@click.option('--vad_filter',   cls=whisperParam, show_default=True, default=True, help='Enable Silero VAD voice to filter out parts without speech')
# Silero Vad params. These are passed through  faster_whisper model.transcribe(), using vad_paramaters Dict
@click.option('--threshold', cls=sileroParam, show_default=True, default=0.5, help=f'{silero_param}Speech threshold. Probabilities ABOVE this value are considered SPEECH. 0.5 is good default.')
@click.option('--min_speech_duration_ms', cls=sileroParam, show_default=True, default=250, help=f'{silero_param}Minimum speech chunk duration in milliseconds. Chunks shorter than this will be discarded.')
@click.option('--max_speech_duration_s', cls=sileroParam, show_default=True, default=float('inf'), help=f'{silero_param}Maximum speech chunk duration in seconds.')
@click.option('--min_silence_duration_ms', cls=sileroParam, show_default=True, default=2000, help=f'{silero_param}Minimum silence duration in milliseconds before separating speech chunks.')
@click.option('--window_size_samples', cls=sileroParam, show_default=True, default=1024, help=f'{silero_param}Number of audio samples per window to be fed to the Silero VAD model.')
@click.option('--speech_pad_ms', cls=sileroParam, default=200, help=f'{silero_param}Padding in milliseconds to be added to each side of the final speech chunks.')
def cli(url, **kwargs):
    main(url, **kwargs)


if __name__ == "__main__":
    cli()
