"""
Make pace calculations based on existing annotations

"""
import pandas as pd
import numpy as np

from common.config_manager import config
from backend.lib.processor import BasicProcessor
from common.lib.user_input import UserInput
from common.lib.exceptions import ProcessorInterruptedException, ProcessorException

from common.lib.helpers import get_ffmpeg_version

__author__ = "Mike Chow"
__credits__ = ["Mike Chow"]
__maintainer__ = "Stijn Peeters"
__email__ = "4cat@oilab.eu"


class VideoSceneFrames(BasicProcessor):
    """
    Video Frame Extracter

    Uses ffmpeg to extract a certain number of frames per second at different sizes and saves them in an archive.
    """
    type = "video-scene-data-translation"  # job type ID
    category = "Visual"  # category
    title = "Translate into video metadata and calculate video pace"  # title displayed in UI
    description = "Using the shot data extracted from the user's selection of shot boundary detection, " \
                  "translate it into metadata for each video. Metadata such as video duration, frame rate will be " \
                  "extracted and video pace calculations will be attached as well. "  \
                  "Additional option available to merge it with the original dataset."  # description displayed in UI
    extension = "csv"  # extension of result file, used internally and in UI

    followups = ["video-timelines"]

    references = [
        "[Redfern: The average shot length as a statistic of film style]" \
        "(https://cinemetrics.uchicago.edu/article/2732e556-9e15-4d0b-b864-e08625fb6acf)"
    ]

    options = {
        "merge_on_parent": {
            "type": UserInput.OPTION_TOGGLE,
            "help": "Append video metadata as annotations to parent dataset",
            "default": True,
            "tooltip": "If enabled, appends calculations for Average Shot Length (ASL) and cuts per minute (Ct/min)" \
                       "to main .csv file."
        }
    }

    @classmethod
    def is_compatible_with(cls, module=None, user=None):
        """
        Determine compatibility

        Compatible with scene data only.

        :param str module:  Module ID to determine compatibility with
        :return bool:
        """
        return module.type in ["video-scene-detector", "preset-scene-timelines"]

    def process(self):
        """
        This takes the data generated from scene boundary detection and appends it as additional video metadata.
        It also makes specific pace calculations for each video.
        """
        # Check processor able to run
        if self.source_dataset.num_rows == 0:
            self.dataset.update_status("No videos from which to extract frames.", is_final=True)
            self.dataset.finish(0)
            return

        self.dataset.update_status("Translating data and processing calculations...")

        # Iterate through source dataset which contains shot information and translate that upwards into metadata
        # for each video
        video_metadata = []

        #  Loop through every dictionary in source dataset
        for shot in self.source_dataset.iterate_items(self):
            # Interrupt message
            if self.interrupted:
                raise ProcessorInterruptedException("Interrupted while detecting video scenes")

            id = shot['url']
            id_exist = False  # Instantiate existence of id check

            # Convert duration data into timedelta objects
            start_time = pd.to_timedelta(shot['start_time'])
            end_time = pd.to_timedelta(shot['end_time'])
            video_duration = pd.to_timedelta(shot['total_video_duration'])

            shot_duration = end_time - start_time

            for video in video_metadata:  # If video ID has already been logged, mark as True
                if video['id'] == id:
                    #  If True, simply append the shot_duration of the shot to the shot_duration_list
                    video['shot_duration_list'].append(shot_duration)
                    id_exist = True

            if not id_exist:
                video_metadata.append({
                    'id': shot['url'],
                    'frame_rate': shot['start_fps'],
                    'shot_count': int(shot['num_scenes_detected']),
                    'total_duration': video_duration,
                    'shot_duration_list': [shot_duration]
                })

        # Loop through each video entry and perform calculations

        for video in video_metadata:
            # Calculate Average Shot Length (ASL)
            video['asl'] = video['total_duration'].total_seconds() / int(video['shot_count'])

            # Calculate Median Shot Length (MSL)
            msl = np.median(video['shot_duration_list'])
            video['msl'] = msl

            # Calculate cuts per min
            # Theory is to understand the number of splits in a video
            video['cuts_per_min'] = (int(video['shot_count']) - 1) / (video['total_duration'].total_seconds() / 60)

        # Finish and submit
        if video_metadata:
            if self.parameters.get("merge_on_parent") == True:  # If it is desired to merge with the parent dataset...

                self.dataset.update_status(
                    'Translation and calculation successful. Merged onto parent dataset')

                # For simplicity sake, convert to pandas and merge on id
                video_list = []
                for video in self.dataset.top_parent().iterate_items(self):
                    video_list.append(video)

                df_video_list = pd.DataFrame(video_list)
                df_video_metadata = pd.DataFrame(video_metadata)
                df_merged = pd.merge(df_video_list, df_video_metadata, on='id', how='left')

                # Convert back to list of dicts
                merged_metadata = df_merged.to_dict('records')

                self.write_csv_items_and_finish(merged_metadata)

            else:
                self.dataset.update_status(
                    'Translation and calculation successful.')
                self.write_csv_items_and_finish(video_metadata)
        else:
            return self.dataset.finish_with_error("Unable to translate video metadata. Error encountered")
