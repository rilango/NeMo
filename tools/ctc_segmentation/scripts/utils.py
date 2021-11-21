# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import logging
import logging.handlers
import multiprocessing
import os
from pathlib import PosixPath
from typing import List, Tuple, Union


# import numpy as np

import logging
import numpy as np

# import for table of character probabilities mapped to time
import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
# from cs import cython_fill_table
from csnewn import cython_fill_table_new

# try:
#     from .ctc_segmentation_dyn import cython_fill_table
# except ImportError:
#     import pyximport
#
#     pyximport.install(setup_args={"include_dirs": np.get_include()})
#     from ctc_segmentation_dyn import cython_fill_table


class CtcSegmentationParameters:
    """Default values for CTC segmentation.

    May need adjustment according to localization or ASR settings.
    The character set is taken from the model dict, i.e., usually are generated with
    SentencePiece. An ASR model trained in the corresponding language and character
    set is needed. If the character set contains any punctuation characters, "#",
    or the Greek char "ε", adapt these settings. If the blank token is not set
    as "_", adjust the setting.
    """

    max_prob = -10000000000.0
    skip_prob = -10000000000.0
    min_window_size = 8000
    max_window_size = 100000
    subsampling_factor = 2
    frame_duration_ms = 20
    score_min_mean_over_L = 30
    space = " "
    blank = "▁"
    self_transition = "ε"
    start_of_ground_truth = "#"
    excluded_characters = ".,-?!:»«;'›‹()"
    char_list = None
    flags=1
    backtrack_from_max_t=False

    @property
    def index_duration_in_seconds(self):
        """Automatically derive index duration from frame duration and subsampling."""
        return self.frame_duration_ms * self.subsampling_factor / 1000


def ctc_segmentation_new(config, lpz, ground_truth):
    """Extract character-level utterance alignments.

    :param config: an instance of CtcSegmentationParameters
    :param lpz: probabilities obtained from CTC output
    :param ground_truth:  ground truth text in the form of a label sequence
    :return:
    """
    blank = 0 #config.blank
    # offset = 0
    audio_duration = lpz.shape[0] * config.index_duration_in_seconds
    logging.info(
        f"CTC segmentation of {len(ground_truth)} chars "
        f"to {audio_duration:.2f}s audio "
        f"({lpz.shape[0]} indices)."
    )
    if len(ground_truth) > lpz.shape[0] and config.skip_prob <= config.max_prob:
        raise AssertionError("Audio is shorter than text!")
    window_size = config.min_window_size
    # Try multiple window lengths if it fails
    while True:
        # Create table of alignment probabilities
        table = np.zeros(
            [min(window_size, lpz.shape[0]), len(ground_truth)], dtype=np.float32
        )
        table.fill(config.max_prob)
        # Use array to log window offsets per character
        offsets = np.zeros([len(ground_truth)], dtype=np.int64)
        # Run actual alignment of utterances
        t, c = cython_fill_table_new(
            table,
            lpz.astype(np.float32),
            np.array(ground_truth, dtype=np.int64),
            offsets,
            config.blank,
            config.flags,
        )
        import pickle
        pickle.dump(table, open("table_new.p", "wb"))
        pickle.dump(offsets, open("offsets_new.p", "wb"))
        print('saved pickles')
        """
        (16000, 41323)
        table[5:10,5:10]
        array([[-9.8981377e+01, -1.0000000e+09, -1.0000000e+09, -1.7474466e+02,
        -1.0000000e+09],
       [-9.8981377e+01, -1.2442617e+02, -1.0000000e+09, -1.6971939e+02,
        -1.9797722e+02],
       [-9.8981377e+01, -1.2276803e+02, -1.4634543e+02, -1.6971939e+02,
        -1.9784351e+02],
       [-9.8981377e+01, -1.2276803e+02, -1.4602182e+02, -1.6971939e+02,
        -1.9416986e+02],
       [-9.8981377e+01, -1.2276803e+02, -1.4602182e+02, -1.6971939e+02,
        -1.9416986e+02]], dtype=float32)
        """
        print('--> t', t, 'new 15924')
        print('--> c', c, 'new 41322')

        if config.backtrack_from_max_t:
            t = table.shape[0] - 1
        logging.debug(
            f"Max. joint probability to align text to audio: "
            f"{table[:, c].max()} at time index {t}"
        )
        # Backtracking
        timings = np.zeros([len(ground_truth)])
        char_probs = np.zeros([lpz.shape[0]])
        state_list = [""] * lpz.shape[0]
        try:
            # Do until start is reached
            while t != 0 or c != 0:
                # Calculate the possible transition probs towards the current cell
                min_s = None
                min_switch_prob_delta = np.inf
                max_lpz_prob = config.max_prob
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = (
                            lpz[t + offsets[c], ground_truth[c, s]]
                            if c > 0
                            else config.max_prob
                        )
                        est_switch_prob = table[t, c] - table[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s
                        max_lpz_prob = max(max_lpz_prob, switch_prob)
                stay_prob = (
                    max(lpz[t + offsets[c], blank], max_lpz_prob)
                    if t > 0
                    else config.max_prob
                )
                est_stay_prob = table[t, c] - table[t - 1, c]
                # Check which transition has been taken
                if abs(stay_prob - est_stay_prob) > min_switch_prob_delta:
                    # Apply reverse switch transition
                    if c > 0:
                        # Log timing and character - frame alignment
                        for s in range(0, min_s + 1):
                            timings[c - s] = (
                                offsets[c] + t
                            ) * config.index_duration_in_seconds
                        char_probs[offsets[c] + t] = max_lpz_prob
                        char_index = ground_truth[c, min_s]
                        state_list[offsets[c] + t] = config.char_list[char_index]
                    c -= 1 + min_s
                    t -= 1 - offset
                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    state_list[offsets[c] + t] = config.self_transition
                    t -= 1
        except IndexError:
            logging.warning(
                "IndexError: Backtracking was not successful, "
                "the window size might be too small."
            )
            window_size *= 2
            if window_size < config.max_window_size:
                logging.warning("Increasing the window size to: " + str(window_size))
                continue
            else:
                logging.error("Maximum window size reached.")
                logging.error("Check data and character list!")
                raise
        break
    return timings, char_probs, state_list

def ctc_segmentation_old(config, lpz, ground_truth):
    """Extract character-level utterance alignments.

    :param config: an instance of CtcSegmentationParameters
    :param lpz: probabilities obtained from CTC output
    :param ground_truth:  ground truth text in the form of a label sequence
    :return:
    """
    blank = 0
    audio_duration = lpz.shape[0] * config.index_duration_in_seconds
    logging.info(
        f"CTC segmentation of {len(ground_truth)} chars "
        f"to {audio_duration:.2f}s audio "
        f"({lpz.shape[0]} indices)."
    )
    if len(ground_truth) > lpz.shape[0] and config.skip_prob <= config.max_prob:
        raise AssertionError("Audio is shorter than text!")
    window_size = config.min_window_size
    # Try multiple window lengths if it fails
    while True:
        # Create table of alignment probabilities
        table = np.zeros(
            [min(window_size, lpz.shape[0]), len(ground_truth)], dtype=np.float32
        )
        table.fill(config.max_prob)
        # Use array to log window offsets per character
        offsets = np.zeros([len(ground_truth)], dtype=np.int)
        # Run actual alignment of utterances
        # import pdb; pdb.set_trace()
        t, c = cython_fill_table(table, lpz.astype(np.float32), np.array(ground_truth), offsets, blank)
        import pickle
        t = 15924
        c = 41322
        table_new = pickle.load( open( "table_new.p", "rb" ) )
        offsets_new = pickle.load(open("offsets_new.p", "rb"))
        print('--> t', t, 'new 15924')
        print('--> c', c, 'new 41322')
        import pdb; pdb.set_trace()

        logging.debug(
            f"Max. joint probability to align text to audio: "
            f"{table[:, c].max()} at time index {t}"
        )
        # Backtracking
        timings = np.zeros([len(ground_truth)])
        char_probs = np.zeros([lpz.shape[0]])
        state_list = [""] * lpz.shape[0]
        try:
            # Do until start is reached
            while t != 0 or c != 0:
                # Calculate the possible transition probs towards the current cell
                min_s = None
                min_switch_prob_delta = np.inf
                max_lpz_prob = config.max_prob
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = (
                            lpz[t + offsets[c], ground_truth[c, s]]
                            if c > 0
                            else config.max_prob
                        )
                        est_switch_prob = table[t, c] - table[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s
                        max_lpz_prob = max(max_lpz_prob, switch_prob)
                stay_prob = (
                    max(lpz[t + offsets[c], blank], max_lpz_prob)
                    if t > 0
                    else config.max_prob
                )
                est_stay_prob = table[t, c] - table[t - 1, c]
                # Check which transition has been taken
                if abs(stay_prob - est_stay_prob) > min_switch_prob_delta:
                    # Apply reverse switch transition
                    if c > 0:
                        # Log timing and character - frame alignment
                        for s in range(0, min_s + 1):
                            timings[c - s] = (
                                offsets[c] + t
                            ) * config.index_duration_in_seconds
                        char_probs[offsets[c] + t] = max_lpz_prob
                        char_index = ground_truth[c, min_s]
                        state_list[offsets[c] + t] = config.char_list[char_index]
                    c -= 1 + min_s
                    t -= 1 - offset
                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    state_list[offsets[c] + t] = config.self_transition
                    t -= 1
        except IndexError:
            logging.warning(
                "IndexError: Backtracking was not successful, "
                "the window size might be too small."
            )
            window_size *= 2
            if window_size < config.max_window_size:
                logging.warning("Increasing the window size to: " + str(window_size))
                continue
            else:
                logging.error("Maximum window size reached.")
                logging.error("Check data and character list!")
                raise
        break
    return timings, char_probs, state_list


def prepare_text(config, text, char_list=None):
    """Prepare the given text for CTC segmentation.

    Creates a matrix of character symbols to represent the given text,
    then creates list of char indices depending on the models char list.

    :param config: an instance of CtcSegmentationParameters
    :param text: iterable of utterance transcriptions
    :param char_list: a set or list that includes all characters/symbols,
                        characters not included in this list are ignored
    :return: label matrix, character index matrix
    """
    if char_list is not None:
        config.char_list = char_list
    logging.debug(f"Blank character: {config.char_list[0]} == {config.blank}")
    ground_truth = config.start_of_ground_truth
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if ground_truth[-1] != config.space:
            ground_truth += config.space
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            if char.isspace():
                if ground_truth[-1] != config.space:
                    ground_truth += config.space
            elif char in config.char_list and char not in config.excluded_characters:
                ground_truth += char
    # Add space to the end
    if ground_truth[-1] != config.space:
        ground_truth += config.space
    utt_begin_indices.append(len(ground_truth) - 1)
    print(ground_truth[:200])
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = max([len(c) for c in config.char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s : i + 1]
            span = span.replace(config.space, config.blank)
            if span in config.char_list:
                ground_truth_mat[i, s] = config.char_list.index(span)
    return ground_truth_mat, utt_begin_indices


def determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text):
    """Utterance-wise alignments from char-wise alignments.

    :param config: an instance of CtcSegmentationParameters
    :param utt_begin_indices: list of time indices of utterance start
    :param char_probs:  character positioned probabilities obtained from backtracking
    :param timings: mapping of time indices to seconds
    :param text: list of utterances
    :return: segments, a list of: utterance start and end [s], and its confidence score
    """

    def compute_time(index, align_type):
        """Compute start and end time of utterance.

        :param index:  frame index value
        :param align_type:  one of ["begin", "end"]
        :return: start/end time of utterance in seconds
        """
        middle = (timings[index] + timings[index - 1]) / 2
        if align_type == "begin":
            return max(timings[index + 1] - 0.5, middle)
        elif align_type == "end":
            return min(timings[index - 1] + 0.5, middle)

    segments = []
    for i in range(len(text)):
        start = compute_time(utt_begin_indices[i], "begin")
        end = compute_time(utt_begin_indices[i + 1], "end")
        start_t = int(round(start / config.index_duration_in_seconds))
        end_t = int(round(end / config.index_duration_in_seconds))
        # Compute confidence score by using the min mean probability
        #   after splitting into segments of L frames
        n = config.score_min_mean_over_L
        if end_t == start_t:
            min_avg = 0
        elif end_t - start_t <= n:
            min_avg = char_probs[start_t:end_t].mean()
        else:
            min_avg = 0
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t : t + n].mean())
        segments.append((start, end, min_avg))
    return segments


def get_segments(
    log_probs: np.ndarray,
    path_wav: Union[PosixPath, str],
    transcript_file: Union[PosixPath, str],
    output_file: str,
    vocabulary: List[str],
    window_size: int = 8000,
    frame_duration_ms: int = 20,
) -> None:
    """
    Segments the audio into segments and saves segments timings to a file

    Args:
        log_probs: Log probabilities for the original audio from an ASR model, shape T * |vocabulary|.
                   values for blank should be at position 0
        path_wav: path to the audio .wav file
        transcript_file: path to
        output_file: path to the file to save timings for segments
        vocabulary: vocabulary used to train the ASR model, note blank is at position 0
        window_size: the length of each utterance (in terms of frames of the CTC outputs) fits into that window.
        frame_duration_ms: frame duration in ms
    """


    with open(transcript_file, "r") as f:
        text = f.readlines()
        text = [t.strip() for t in text if t.strip()]

    # add corresponding original text without pre-processing
    transcript_file_no_preprocessing = transcript_file.replace('.txt', '_with_punct.txt')
    if not os.path.exists(transcript_file_no_preprocessing):
        raise ValueError(f'{transcript_file_no_preprocessing} not found.')

    with open(transcript_file_no_preprocessing, "r") as f:
        text_no_preprocessing = f.readlines()
        text_no_preprocessing = [t.strip() for t in text_no_preprocessing if t.strip()]

    # add corresponding normalized original text
    transcript_file_normalized = transcript_file.replace('.txt', '_with_punct_normalized.txt')
    if not os.path.exists(transcript_file_normalized):
        raise ValueError(f'{transcript_file_normalized} not found.')

    with open(transcript_file_normalized, "r") as f:
        text_normalized = f.readlines()
        text_normalized = [t.strip() for t in text_normalized if t.strip()]

    if len(text_no_preprocessing) != len(text):
        raise ValueError(f'{transcript_file} and {transcript_file_no_preprocessing} do not match')

    if len(text_normalized) != len(text):
        raise ValueError(f'{transcript_file} and {transcript_file_normalized} do not match')

    config = CtcSegmentationParameters()
    config.char_list = vocabulary
    config.min_window_size = window_size
    config.frame_duration_ms = frame_duration_ms
    config.blank = config.space
    config.subsampling_factor = 2
    ground_truth_mat, utt_begin_indices = prepare_text(config, text)
    # print("OLD package")
    # print(ground_truth_mat.shape)
    # _print(ground_truth_mat, vocabulary)
    print('+'*40)

    config_new = CtcSegmentationParametersNEW()
    excluded_characters_new = ".,»«•❍·"
    excluded_characters_old = ".,-?!:»«;'›‹()"
    config_new.excluded_characters = excluded_characters_old
    config_new.char_list = vocabulary
    config_new.min_window_size = window_size
    config_new.blank = vocabulary.index(config.space)
    # config_new.frame_duration_ms = frame_duration_ms
    # config_new.subsampling_factor = 2
    config_new.index_duration = 0.04

    ground_truth_mat_new, utt_begin_indices_new = prepare_textNEW(config_new, text)

    print("NEW package")
    print(ground_truth_mat_new.shape)
    _print(ground_truth_mat_new, vocabulary)
    print(ground_truth_mat_new[:10])
    import pdb; pdb.set_trace()
    """
    {'excluded_characters': ".,-?!:»«;'›‹()", 
    'char_list': ['ε', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"], 
    'min_window_size': 16000, 'blank': 1, 'index_duration': 0.04}
    (41323, 1)
    [[-1]
     [ 1]
     [ 9]
     [ 6]
     [13]
     [13]
     [16]
     [ 1]
     [ 2]
     [15]]
    
    []
    [' ']
    ['h']
    ['e']
    ['l']
    ['l']
    ['o']
    [' ']

    """

    # for i in range(ground_truth_mat.shape[0]):
    #     if ground_truth_mat[i] != ground_truth_mat_new[i]:
    #         print(i)
    #         _print(ground_truth_mat_new[i - 5:i + 5], vocabulary)
    #         print("-"*40)
    #         _print(ground_truth_mat_new[i-5:i+5], vocabulary)
    #         import pdb; pdb.set_trace()
    #         print()




    # logging.debug(f"Syncing {transcript_file}")
    # logging.debug(
    #     f"Audio length {os.path.basename(path_wav)}: {log_probs.shape[0]}. "
    #     f"Text length {os.path.basename(transcript_file)}: {len(ground_truth_mat)}"
    # )
    #
    config_new.blank=0
    timings, char_probs, char_list = ctc_segmentation_new(config_new, log_probs, ground_truth_mat_new)

    # utt_begin_indices = utt_begin_indices_new
    # # config_new.blank_transition_cost_zero = True
    # config_new.backtrack_from_max_t = False
    # config.frame_duration_ms=20
    # config.subsampling_factor=2
    # config.flags=2
    #
    # # try oold config and new matrcix with new C+
    # config.blank = 0
    # timings, char_probs, char_list = cs.ctc_segmentation(config, log_probs, ground_truth_mat)
    # import pdb;
    # pdb.set_trace()
    segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
    write_output(output_file, path_wav, segments, text, text_no_preprocessing, text_normalized)
    for i, (word, segment) in enumerate(zip(text, segments)):
        if i < 5:
            logging.debug(f"{segment[0]:.2f} {segment[1]:.2f} {segment[2]:3.4f} {word}")

class CtcSegmentationParametersNEW:
    """Default values for CTC segmentation.

    May need adjustment according to localization or ASR settings.
    The character set is taken from the model dict, i.e., usually are generated
    with SentencePiece. An ASR model trained in the corresponding language and
    character set is needed. If the character set contains any punctuation
    characters, "#", the Greek char "ε", or the space placeholder, adapt
    these settings.
    """

    max_prob = -10000000000.0
    skip_prob = -10000000000.0
    min_window_size = 8000
    max_window_size = 100000
    index_duration = 0.025
    score_min_mean_over_L = 30
    space = "·"
    blank = 0
    replace_spaces_with_blanks = False
    blank_transition_cost_zero = False
    preamble_transition_cost_zero = True
    backtrack_from_max_t = False
    self_transition = "ε"
    start_of_ground_truth = "#"
    excluded_characters = ".,»«•❍·"
    tokenized_meta_symbol = "▁"
    char_list = None
    # legacy Parameters (will be ignored in future versions)
    subsampling_factor = None
    frame_duration_ms = None

    @property
    def index_duration_in_seconds(self):
        """Derive index duration from frame duration and subsampling.

        This value can be fixed by setting ctc_index_duration, which causes
        frame_duration_ms and subsampling_factor to be ignored.

        Legacy function. This function will be removed in later versions
        and replaced by index_duration.
        """
        if self.subsampling_factor and self.frame_duration_ms:
            t = self.frame_duration_ms * self.subsampling_factor / 1000
        else:
            t = self.index_duration
        return t

    @property
    def flags(self):
        """Get configuration flags to pass to the table_fill operation."""
        flags = int(self.blank_transition_cost_zero)
        flags += 2 * int(self.preamble_transition_cost_zero)
        return flags

    def update_excluded_characters(self):
        """Remove known tokens from the list of excluded characters."""
        self.excluded_characters = "".join(
            [
                char
                for char in self.excluded_characters
                if True not in [char == j for j in self.char_list]
            ]
        )
        logging.debug(f"Excluded characters: {self.excluded_characters}")

    def __init__(self, **kwargs):
        """Set all parameters as attribute at init."""
        self.set(**kwargs)

    def set(self, **kwargs):
        """Update CtcSegmentationParameters.

        Args:
            **kwargs: Key-value dict that contains all properties
                with their new values. Unknown properties are ignored.
        """
        for key in kwargs:
            if (
                not key.startswith("_")
                and hasattr(self, key)
                and kwargs[key] is not None
            ):
                setattr(self, key, kwargs[key])

    def __repr__(self):
        """Print all attribute as dictionary."""
        output = "CtcSegmentationParameters( "
        for attribute in self.__dict__.keys():
            value = self.__dict__[attribute]
            output += f"{attribute}={value}, "
        output += ")"
        return output

def prepare_textNEW(config, text, char_list=None):
    """Prepare the given text for CTC segmentation.

    Creates a matrix of character symbols to represent the given text,
    then creates list of char indices depending on the models char list.

    :param config: an instance of CtcSegmentationParameters
    :param text: iterable of utterance transcriptions
    :param char_list: a set or list that includes all characters/symbols,
                        characters not included in this list are ignored
    :return: label matrix, character index matrix
    """
    # temporary compatibility fix for previous espnet versions
    if type(config.blank) == str:
        config.blank = 0
    if char_list is not None:
        config.char_list = char_list
    blank = config.char_list[config.blank]
    ground_truth = config.start_of_ground_truth
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth.endswith(config.space):
            ground_truth += config.space
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            if char.isspace() and config.replace_spaces_with_blanks:
                if not ground_truth.endswith(config.space):
                    ground_truth += config.space
            elif char in config.char_list and char not in config.excluded_characters:
                ground_truth += char
    # Add space to the end
    if not ground_truth.endswith(config.space):
        ground_truth += config.space
    logging.debug(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = max([len(c) for c in config.char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s : i + 1]
            span = span.replace(config.space, blank)
            if span in config.char_list:
                char_index = config.char_list.index(span)
                ground_truth_mat[i, s] = char_index
    return ground_truth_mat, utt_begin_indices

def write_output(
    out_path: str,
    path_wav: str,
    segments: List[Tuple[float]],
    text: str,
    text_no_preprocessing: str,
    text_normalized: str,
    stride: int = 2,
):
    """
    Write the segmentation output to a file

    out_path: Path to output file
    path_wav: Path to the original audio file
    segments: Segments include start, end and alignment score
    text: Text used for alignment
    text_no_preprocessing: Reference txt without any pre-processing
    text_normalized: Reference text normalized
    stride: Stride applied to an ASR input
    """
    # Uses char-wise alignments to get utterance-wise alignments and writes them into the given file
    with open(str(out_path), "w") as outfile:
        outfile.write(str(path_wav) + "\n")

        for i, (start, end, score) in enumerate(segments):
            outfile.write(
                f'{start/stride} {end/stride} {score} | {text[i]} | {text_no_preprocessing[i]} | {text_normalized[i]}\n'
            )

def _print(ground_truth_mat, vocabulary, limit=20):
    """Prints transition matrix"""
    chars = []
    for row in ground_truth_mat:
        chars.append([])
        for ch_id in row:
            if ch_id != -1:
                chars[-1].append(vocabulary[int(ch_id)])

    for x in chars[:limit]:
        print(x)

#####################
# logging utils
#####################
def listener_configurer(log_file, level):
    root = logging.getLogger()
    h = logging.handlers.RotatingFileHandler(log_file, 'w')
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    ch = logging.StreamHandler()
    root.addHandler(h)
    root.setLevel(level)
    root.addHandler(ch)


def listener_process(queue, configurer, log_file, level):
    configurer(log_file, level)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.setLevel(logging.INFO)
            logger.handle(record)  # No level or filter logic applied - just do it!

        except Exception:
            import sys
            import traceback

            print('Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def worker_configurer(queue, level):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(level)


def worker_process(
    queue, configurer, level, log_probs, path_wav, transcript_file, output_file, vocabulary, window_len
):
    configurer(queue, level)
    name = multiprocessing.current_process().name
    innerlogger = logging.getLogger('worker')
    innerlogger.info(f'{name} is processing {path_wav}, window_len={window_len}')
    get_segments(log_probs, path_wav, transcript_file, output_file, vocabulary, window_len)
    innerlogger.info(f'{name} completed segmentation of {path_wav}, segments saved to {output_file}')
