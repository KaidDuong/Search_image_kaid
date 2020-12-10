

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from app.mrcnn.config import Config
from app.mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class GraphicalObjectConfig(Config):
    """Configuration for training on the graphical object in document  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "graphical_object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + table, figure, natural image, logo, and signature

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class GraphicalObjectDataset(utils.Dataset):

    def load_graphical_object(self, dataset_dir, subset):
        """Load a subset of the graphical object in document images dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("graphical_object", 1, "table")
        self.add_class("graphical_object", 2, "figure")
        self.add_class("graphical_object", 3, "natural_image")
        self.add_class("graphical_object", 4, "signature")
        self.add_class("graphical_object", 5, "logo")
        # Train or validation dataset?
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        data = json.load(open(os.path.join(dataset_dir, "data.json")))

        # Add images
        for item in data:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            ann = item['annotation']
            objects = [{k :v for (k,v) in obj.items() if k in ['name', 'bndbox'] } for obj in ann['object']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, item['file_name'])
            image = skimage.io.imread(image_path)
            height = int(ann['size']['height'])
            width = int(ann['size']['width'])

            self.add_image(
                "graphical_object",
                image_id=item['file_name'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                objects = objects) 

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "graphical_object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        # create one array for all masks, each on a different channel
        info = self.image_info[image_id]
        objects = info['objects']
        count = len(objects)

        # create one array for all masks, each on a different channel
        masks = np.zeros([info["height"], info["width"], count],
                        dtype=np.uint8)
        
        # create masks
        for i in range(count):
            box = objects[i]['bndbox']
            row_s, row_e = int(box['ymin']), int(box['ymax'])
            col_s, col_e = int(box['xmin']), int(box['xmax'])
            masks[row_s:row_e, col_s:col_e, i] = 1

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(obj['name']) for obj in objects])
        return masks, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "graphical_object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = GraphicalObjectDataset()
    dataset_train.load_graphical_object(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GraphicalObjectDataset()
    dataset_val.load_graphical_object(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')





############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect graphical object in document images.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the Graphical object in document images dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GraphicalObjectConfig()
    else:
        class InferenceConfig(GraphicalObjectConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
