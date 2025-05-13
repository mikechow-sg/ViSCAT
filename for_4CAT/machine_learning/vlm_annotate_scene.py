"""
Script that takes in a user input for a prompt and runs a dataset of scene thumbnails through a vlm model
"""
import os
import pandas as pd
import re
import json
import lmstudio as lms

from common.config_manager import config
from backend.lib.processor import BasicProcessor
from common.lib.user_input import UserInput
from common.lib.exceptions import ProcessorInterruptedException, ProcessorException

from common.lib.helpers import get_ffmpeg_version

__author__ = "Mike Chow"
__credits__ = ["Mike Chow"]
__maintainer__ = "Stijn Peeters"
__email__ = "4cat@oilab.eu"


class VlmSceneAnnotation(BasicProcessor):
    """
    VLM Scene Thumbnail Annotator

    Takes in a user prompt and runs it through a dataset of scene thumbnails through a VLM model and appends it to
    the scene metadata.
    """
    type = "vlm-annotate-scenes"  # job type ID
    category = "Visual"  # category
    title = "Annotate scenes using a visual-language model (VLM)"  # title displayed in UI
    description = (
        'Uses a visual language model to annotate scenes in a video. '
        'The model will be used to predict an annotation for the scene thumbnail for each scene in the video dataset. '
        'and the resulting annotations will be appended to the video metadata. \n'
        'CAUTION: You need LM Studio to be installed and running with the desired model loaded to run this processor. '
        'Please make sure that \'Enable CORS\' is turned on in the LM Studio settings. '
    )  # description displayed in UI
    extension = "csv"  # extension of result file, used internally and in UI

    followups = ["vlm-annotate-scenes"]

    references = ['[Link to be updated](https://cinemetrics.uchicago.edu/article/2732e556-9e15-4d0b-b864-e08625fb6acf)']

    options = {
        "model_name": {
            "type": UserInput.OPTION_CHOICE,
            "default": "qwen2-vl-7b-instruct",
            "options": {
                "qwen2-vl-7b-instruct": "Qwen2-VL-7B-Instruct",
            },
            "help": "Name of desired VLM"
        },
            "prompt": {
                "type": UserInput.OPTION_TEXT_LARGE,
                "help": "Prompt for the visual language model",
                "default": "You are an expert annotator in videos and cinematography. You are given a scene thumbnail "
                           "of a longer video. Your job is to... . Only return the classification.",
                "tooltip": "Prompt for the visual language model"
        },
            "column_name": {
                "type": UserInput.OPTION_TEXT,
                "default": "annotation",
                "help": "Name of the new column to be added to the scene metadata",
            },
    }

    @classmethod
    def is_compatible_with(cls, module=None, user=None):
        """
        Determine compatibility

        Compatible ONLY with after scene data and thumbnail collection.

        :param str module:  Module ID to determine compatibility with
        :return bool:
        """
        return module.type in ["video-scene-frames", "preset-scene-timelines", "vlm-annotate-scenes"]

    def process(self):

        # Get scene metadata
        genealogy_list = self.dataset.get_genealogy()
        scene_dataset = None
        for past_dataset in genealogy_list:
            if past_dataset.type == "video-scene-detector":
                scene_data = past_dataset.get_results_path()
                scene_dataset = past_dataset  # List of dictionaries
                break

        if not scene_dataset: # No video scene metadata? How can it be!
            return self.dataset.finish_with_error("No video scene metadata found")

        scene_list = []
        for scene in scene_dataset.iterate_items(self):
            scene_list.append(scene)

        # Prepare for prompting
        global prompt, model_name, column_name
        prompt = self.parameters.get("prompt")
        model_name = self.parameters.get("model_name")
        column_name = self.parameters.get("column_name")

        # Loop through archive of images
        ## Code that checks for zip file of images so that its not reliant on source dataset...
        iterator = self.iterate_archive_contents(self.source_file)
        looping = True

        scene_count = 0
        while looping:
            try:
                file_path = next(iterator)

            except StopIteration:
                looping = False
                break
            except Exception as e:
                print(f'Error reading file: {e}')

            # there is a metadata file, always read first, which we can use (Just in case)
            if file_path.name == ".metadata.json":
                with file_path.open() as infile:
                    metadata = json.load(infile)
                    continue

            # skip if file is a real file but not an image
            if looping and file_path.suffix not in (".jpeg", ".jpg", ".png", ".gif"):
                continue


            # Prompt through VLM
            scene_count += 1
            self.dataset.update_status(f'{file_path.name} read by VLM ({scene_count}/{len(scene_list)})')
            self.dataset.update_progress(scene_count/len(scene_list))
            try:
                # Obtain prediction
                self.dataset.update_status(f'Predicting scene annotation for {file_path.name}')
                prediction = self.get_vlm_prediction(file_path)

                # Add prediction to the correct shot
                # Loop through scene_metadata
                match = False
                for scene_dict in scene_list:

                    # scene_dict['id'] example '7295526580741229825.mp4_scene_1'
                    cleaned_id = re.sub(r'\.mp4', '', scene_dict['id'])
                    # cleaned_id example '7295526580741229825_scene_1'
                    target_id = f'{cleaned_id}.jpeg'

                    # Find match with file_path.name
                    if target_id == file_path.name:
                        scene_dict[column_name] = prediction
                        match = True
                        break

                # What happens if there is no match in the whole list?
                if not match:
                    self.dataset.update_status(f"No match found for {file_path.name}")

            except Exception as e:
                self.dataset.update_status(f'Error in VLM prediction: {e}')

        # Finish up
        if not scene_list:
            return self.dataset.finish_with_error("No files found in archive")
        else:
            self.write_csv_items_and_finish(scene_list)

    def get_vlm_prediction(self, image_path):

        """
        :param image_path: (str) path to the image file.
        :return: prediction.content (str) prediction based on prompt
        """

        with lms.Client(api_host="host.docker.internal:1234") as client:
            image_handle = client.prepare_image(image_path)  # Convert to image_handle
            model = client.llm.model(model_name)
            chat = lms.Chat()
            chat.add_user_message(prompt, images=[image_handle])
            prediction = model.respond(chat)

        self.dataset.update_status(f'Prediction for image {image_path.name}: {prediction.content}')

        return prediction.content
