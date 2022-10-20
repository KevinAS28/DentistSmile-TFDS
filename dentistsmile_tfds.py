"""dentistsmile_tfds dataset."""

import tensorflow_datasets as tfds
import os, random, re
import urllib.request


tfds.core.utils.gcs_utils._is_gcs_disabled = True

'''
to do:
trainval.txt -> identifikasi penomoran
original
true_mask
download url?
'''

# TODO(dentistsmile_tfds): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(dentistsmile_tfds): BibTeX citation
_CITATION = """
"""

OFFLINE = False
DATASET_BASE_DIR = '/home/kevin/dentistsmile/dentistsmile_tfds/dataset'


_POSE_CLASSES = ['A', 'B', 'C', 'D', 'E']
_BASE_URL = 'http://35.225.193.202:9999'

class DentistsmileTfds(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dentistsmile_tfds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }  
  MANUAL_DOWNLOAD_INSTRUCTIONS = f'currently from {DATASET_BASE_DIR}'


#   def __init__(self, manual_offline=True, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     self.manual_offline = manual_offline

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    # TODO(dentistsmile_tfds): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(names=_POSE_CLASSES),
            "file_name": tfds.features.Text(),            
            "true_mask": tfds.features.Image(shape=(None, None, 1), use_colormap=True)            
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='',
        citation=_CITATION,
    )
  

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(dentistsmile_tfds): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    if OFFLINE:
        path = {'original_all': 'Original All', 'true_mask': 'segmentation_true_masks'}
        path = {key:os.path.join(dl_manager.manual_dir, 'dataset', value) for key, value in path.items()}
    else:
        path = dl_manager.download_and_extract({    
            'original_all': _BASE_URL + '/file_server0/download/dentistsmile_images.tar.xz',
            'true_mask': _BASE_URL + '/file_server0/download/dentistsmile_annotations.tar.xz'
        })    
    # annotations_path_dir = os.path.join(_DATASET_BASE_DIR, path['true_masks'], "annotations")
    # TODO(dentistsmile_tfds): Returns the Dict[split names, Iterator[Key, Example]]
    # Setup train and test splits
    train_split = tfds.core.SplitGenerator(
        name="train",
        gen_kwargs={
            "original_image_dir_path":
                path['original_all'],
            "true_mask_dir_path":
                path['true_mask'],
            # "images_list_file":
            #     os.path.join(annotations_path_dir, "trainval.txt"),
        },
    ) 

    test_split = tfds.core.SplitGenerator(
        name="test",
        gen_kwargs={
            "original_image_dir_path":
                path['original_all'],
            "true_mask_dir_path":
                path['true_mask'],
            # "images_list_file":
            #     os.path.join(annotations_path_dir, "trainval.txt"),
        },
    ) 

    return [train_split, test_split]
    

  def _generate_examples(self, original_image_dir_path, true_mask_dir_path):
    """Yields examples."""
    # TODO(dentistsmile_tfds): Yields (key, example) tuples from the dataset
    # for f in path.glob('*.jpeg'):
    #   yield 'key', {
    #       'image': f,
    #       'label': 'yes',
    #   }
    for img_mask_dir in os.listdir(true_mask_dir_path):
        img_id, pose, extension = re.search(r'([\d]{4,4})([A-Z]{1,1}).([A-Z|a-z]{1,4})', img_mask_dir).groups()
        img_mask = os.path.join(true_mask_dir_path, img_mask_dir, f'{img_id+pose}_filled_line.png')
        ori_img = os.path.join(original_image_dir_path, img_mask_dir)

        record = {
            "image": ori_img,
            "label": pose,
            "file_name": img_mask_dir,
            # "segmentation_mask": img_mask,
            "true_mask": img_mask
        }

        yield img_mask_dir, record
    
    # for 
    