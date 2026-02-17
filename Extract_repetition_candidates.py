import pandas as pd
from praatio import textgrid 
import numpy as np
import tgt
import regex as re
from typing import List, Optional, Literal
from itertools import product

class textgrid_preprocessing:
    """ 
    Class for preprocessing (multi-speaker) Textgrids and extracting speech intervals candidate to be source-repetition pairs.
    Args:
        textgrid_file (str): Path to the TextGrid file containing speech intervals.
        child_tier (str): Tier name for speech intervals of the target child of whom you want to examine repetitive speech.
        non_speech_tiers (Optional[List[str]]): Names of tiers that should not be considered 
                                                as possible sources for repetitive speech (e.g. annotation tiers).
        filter_unintelligible (Optional[str]): Regular expression that can be used to exclude intervals that only contains unintelligible speech.
    """ 


    def __init__(
        self,
        textgrid_file: str,
        child_tier: str= None,
        non_speech_tiers: Optional[List[str]]= None,
        filter_unintelligible: Optional[str]= None

    ):


        self.textgrid_file= textgrid_file
        self.child_tier= child_tier
        self.non_speech_tiers= non_speech_tiers
        self.filter_unintelligible= filter_unintelligible

        self.input_tg= textgrid.openTextgrid(self.textgrid_file, includeEmptyIntervals= True)

        self.child_intervals= None
        self.other_speaker_intervals= None


    def get_speech_intervals(self, tier, empty=False):
        """Takes a tier name as input and returns either the timestamps of the non-empty intervals ('empty= False')
        or those of the empty intervals ('empty=True') of that tier,
        filtering out unintelligible utterances if a regular expression is provided for this purpose"""
        entries= self.input_tg.getTier(tier).entries
        intervals= {}
        for entry in entries:
            if  (entry.label and empty==False and not self.filter_unintelligible) or (not entry.label and empty==True)\
                or (entry.label and empty==False and self.filter_unintelligible and not re.match(self.filter_unintelligible, entry.label)):
                intervals[(entry.start, entry.end)]= entry.label.translate(str.maketrans('','', '+?!,/.()[]')) # Remove punctuation
        return intervals

    def get_speaker_dictionaries(self):
        """Builds two speaker dictionaries: (i) for the reference child and (ii) for all other speakers,
        using the function `get_speech_intervals`. The second dictionary contains an additional hierarchical level,
        i.e. the tier name of the speaker.
        """

        self.child_intervals= {}
        self.other_speaker_intervals= {}

        all_tiers= self.input_tg.tierNames
        for tier in all_tiers:
            if tier== self.child_tier:
                self.child_intervals= self.get_speech_intervals(tier)
            elif not self.non_speech_tiers or tier not in self.non_speech_tiers:
                self.other_speaker_intervals[tier]= self.get_speech_intervals(tier)


        return self.child_intervals, self.other_speaker_intervals


    def get_repetition_candidates(self,
                                  repetition_type: Literal["direct", "self-repetition"],
                                  time_distance: Optional[int] = None,
                                 output: Literal['DataFrame','dictionary']= 'DataFrame'):

        """
        Extracts intervals of candidates for source & repetitive pairs in a DataFrame or dictionary containing name of the file, type of repetition,
        speaker compared with the reference child (set to 'same speaker' for self-repetition), and timestamps and transcription of speech intervals.
        Args:
            repetition_type: Type of repetition, either 'direct' or 'self-repetition'. 
                For definitions, see https://osf.io/83bvw/overview?view_only=898ad193f8e54c62b730346238b63cf8.
            time_distance (Optional[int]): Maximal time distance (s) between a source & repetitive candidate pair for direct repetition.
            output (Literal['DataFrame','dictionary']): Chosen data type of output.

        """


        if repetition_type== 'self-repetition' and time_distance:
            raise ValueError('time_distance can only be specified for direct repetitions')

        if output not in ['DataFrame','dictionary']:
            raise ValueError("Output must be one of ['DataFrame','dictionary']")


        if time_distance is None:
                time_distance = 10 if repetition_type == "direct" else None


        rep_dict = {}
        keys = ['file', 'repetition_type', 'comparison_speaker', 'source_interval', 'rep_interval', 'source_speech', 'rep_speech'] 

        if repetition_type== 'direct':

            if not self.child_intervals or self.other_speaker_intervals:
                self.child_intervals, self.other_speaker_intervals= self.get_speaker_dictionaries()


            i = 0
            # Comparison between child and other speaker
            for s2 in self.other_speaker_intervals.keys():
                for (start_child, end_child), child_speech in self.child_intervals.items():
                    for (start_s2, end_s2), s2_speech in self.other_speaker_intervals[s2].items():
                        if 0 < start_child - start_s2 <= time_distance:


                            values = [self.textgrid_file, repetition_type, s2, (start_s2, end_s2), (start_child, end_child),
                                      self.other_speaker_intervals[s2][start_s2, end_s2], self.child_intervals[start_child, end_child]]


                            rep_dict[i] = {key: value for key, value in zip(keys, values)}
                            i += 1

        else: # Self-repetition

            if not self.child_intervals:
                self.child_intervals= self.get_speech_intervals(self.child_tier)


            i = 0
            for (start_source, end_source), source_speech in self.child_intervals.items():
                for (start_rep, end_rep), rep_speech in self.child_intervals.items():
                    if start_source < start_rep:

                        values = [self.textgrid_file, repetition_type, 'same speaker', (start_source, end_source), (start_rep, end_rep),
                                  self.child_intervals[start_source, end_source], self.child_intervals[start_rep, end_rep]]

                        rep_dict[i] = {key: value for key, value in zip(keys, values)}

                        i += 1

        result= rep_dict if output== 'dictionary' else pd.DataFrame(rep_dict).T

        return result



# Example use: 

#filter_unintelligible= '^(xxx|yyy)\s?(\[.+\])?\.?$'
#textgrid_file= 'My_example.Textgrid'
#non_speech_tiers= ['Ignore_this_tier','Ignore_this_one_too']

#test= textgrid_preprocessing(textgrid_file= textgrid_file, child_tier='Transcription', non_speech_tiers= non_speech_tiers,
                 #filter_unintelligible= filter_unintelligible)

#rep_df= test.get_repetition_candidates(repetition_type='direct')