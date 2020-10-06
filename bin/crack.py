import os
import sys

import math
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage
import skimage.io

import datetime


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

#import Mask RCNN
sys.path.append(ROOT_DIR)   # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
from mrcnn import model as modellib

# Directory to save logs and trained model_types
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


## configurations

class CracksConfig(Config):
    NAME = "cracks"

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 cracks

    # Number of training and validation steps per epoch
    #STEPS_PER_EPOCH = (240 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    #VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_SCALE = 0.0

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    LEARNING_RATE = 0.001

    ########################################################
    # Select configuration parameters depends on the paper
    # "Instant bridge visual inspection using an unmanned ~"
    isGeneral = False

    if isGeneral:
        # Batch size 1
        GPU_COUNT = 4
        IMAGES_PER_GPU = 1

        # Steps per epoch = 500
        STEPS_PER_EPOCH = 500

        # weight decay = 0.01
        WEIGHT_DECAY = 0.01

        # Multi-stage is unset middle on the execute code
        # No. of epoch = 100 is set in the train function
        # validation steps = 50
        VALIDATION_STEPS = 50
        # Learning rate = 0.001
        LEARNING_RATE = 0.001
    else : # Proposed configuration

        # Batch size 4
        GPU_COUNT = 4
        IMAGES_PER_GPU = 1

        # Steps per epoch = 1000
        STEPS_PER_EPOCH = 1000

        # weight decay = 0.001
        WEIGHT_DECAY = 0.001

        # Multi-stage is set middle on the execute code
        # Layer is stage 1 "heads" / stage 2 "all"
        # No. of epoch      Stage-1  = 100 / Stage-2 200
        # validation steps  Stage-1  = 50  / Stage-2 200
        VALIDATION_STEPS = 50
        # Learning rate     Stage-1  = 0.001  / Stage-2 0.0001
        LEARNING_RATE = 0.001

###
# dataset
###

class CracksDataset(utils.Dataset):

    # in this case dataset_dir is /data/datasets/crack/"dataset name"
    def load_cracks(self, dataset_dir, subset):

        # Add NUM_CLASSES
        self.add_class("cracks", 1, "crack")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        image_dir = os.path.join(dataset_dir, "image")

        # annotations
        image_list = os.listdir(image_dir)

        for a in image_list:
            image_path = os.path.join(image_dir, a)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            #print(image.shape[:2])

            self.add_image(
                "cracks",
                image_id=a, # use file name as a unique image # id
                path=image_path,
                width=width, height=height,
                #polygons="polygons",
                #subset=subset,
                dataset_dir=dataset_dir)
                # Add images

        # Add images
        # for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.

            # if type()


    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        returns:
            masks : A bool array of shape [height, width, instance count]
            with one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cracks":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        dataset_dir = info["dataset_dir"]
        mask_dir = os.path.join(dataset_dir, "mask")


        # change the extension jpg to rpn_graph
        mask_name = os.path.splitext(info["id"])
        mask_path = os.path.join(mask_dir, mask_name[0] + ".png")

        mask = np.array(skimage.io.imread(mask_path))

        mask = mask.reshape(mask.shape[0], mask.shape[-1], 1)
        #print("In Crack.py, mask shape is : {}".format(mask.shape))

        mask = mask > 128


        #mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                dtype=np.uint8)
        #for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        #    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #    mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cracks":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


###
#
###

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CracksDataset()
    dataset_train.load_cracks(args.dataset, "train")
    dataset_train.prepare()

    print("Image Count of dataset_train: {}".format(len(dataset_train.image_ids)))
    print("Class Count of dataset_train: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # Validation dataset
    dataset_val = CracksDataset()
    dataset_val.load_cracks(args.dataset, "val")
    dataset_val.prepare()

    print("Image Count of dataset_val: {}".format(len(dataset_val.image_ids)))
    print("Class Count of dataset_val: {}".format(dataset_val.num_classes))
    for i, info in enumerate(dataset_val.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        #print("Running on {}".format(args.image))
        print("Running on {}".format(image_path))
        # Read image
        #image = skimage.io.imread(args.image)
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)



###
# training
###

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect crack.')

    parser.add_argument("command", metavar="<command>", help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/data/datasets/crack/crack500/",
                        help='Directory of the Crack dataset')
    parser.add_argument('--weights', required=True,
                        metavar="../mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default="../logs",
                        metavar="../logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="../images", # metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="../videos", # metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()


    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CracksConfig()
    else:
        class InferenceConfig(CracksConfig):
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
        weights_path = COCO_MODEL_PATH
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

    # set multi-GPU
    #from keras.utils import multi_gpu_model
    #model = multi_gpu_model(model,gpus=4)

    # Train or evaluate
    if args.command == "train":
        train(model)
        model_path = os.path.join(MODEL_DIR, "mask_rcnn_cracks.h5")
        model.keras_model.save_weights(model_path)

    elif args.command == "splash":
        targetImage_list = os.listdir(args.image)
        for aa in targetImage_list:
            targetImage_path = os.path.join(args.image, aa)
            print("path : {}".format, targetImage_path)
            detect_and_color_splash(model, image_path=targetImage_path,
        #detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
