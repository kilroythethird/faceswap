#!/usr/bin/env python3
""" Base class for Face Detector plugins
    Plugins should inherit from this class

    See the override methods for which methods are
    required.

    For each source frame, the plugin must pass a dict to finalize containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of dicts containing bounding box points>}}

    - Use the function self.to_bounding_box_dict(left, right, top, bottom) to define the dict
    """

import logging
import os
import traceback
from io import StringIO

import cv2
import numpy as np

from lib.gpu_stats import GPUStats
from lib.utils import rotate_landmarks, GetModel
from plugins.extract._config import Config
from lib.queue_manager import queue_manager
from lib.multithreading import FSThread
import queue
import threading

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name, configfile=None):
    """ Return the config for the requested model """
    return Config(plugin_name, configfile=configfile).config_dict


class Detector():
    """ Detector object """
    def __init__(self, loglevel, configfile=None,  # pylint:disable=too-many-arguments
                 git_model_id=None, model_filename=None, rotation=None, min_size=0):
        logger.debug("Initializing %s: (loglevel: %s, configfile: %s, git_model_id: %s, "
                     "model_filename: %s, rotation: %s, min_size: %s)",
                     self.__class__.__name__, loglevel, configfile, git_model_id,
                     model_filename, rotation, min_size)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]), configfile=configfile)
        self.loglevel = loglevel
        self.rotation = self.get_rotation_angles(rotation)
        self.min_size = min_size
        self.parent_is_pool = False
        self.init = None
        self.error = None

        # The input and output queues for the plugin.
        # See lib.queue_manager.QueueManager for getting queues
        self.queues = {"in": None, "out": None}

        #  Path to model if required
        self.model_path = self.get_model(git_model_id, model_filename)

        # Target image size for passing images through the detector
        # Set to tuple of dimensions (x, y) or int of pixel count
        self.target = None

        # Approximate VRAM used for the set target. Used to calculate
        # how many parallel processes / batches can be run.
        # Be conservative to avoid OOM.
        self.vram = None

        # Set to true if the plugin supports PlaidML
        self.supports_plaidml = False

        # For detectors that support batching, this should be set to
        # the calculated batch size that the amount of available VRAM
        # will support. It is also used for holding the number of threads/
        # processes for parallel processing plugins
        self.batch_size = 1
        self._threads = list()
        self.got_input_eof = False
        logger.debug("Initialized _base %s", self.__class__.__name__)

    def detect_and_raise_errors(self):
        for thread in self._threads:
            thread.check_and_raise_error()

    # <<< OVERRIDE METHODS >>> #
    def initialize(self, *args, **kwargs):
        """ Inititalize the detector
            Tasks to be run before any detection is performed.
            Override for specific detector """
        logger.info("initialize %s (PID: %s, args: %s, kwargs: %s)",
                     self.__class__.__name__, os.getpid(), args, kwargs)
        self.event = self.init = threading.Event() #kwargs.get("event", False)
        #self.error = kwargs.get("error", False)
        self.queues["in"] = kwargs["in_queue"]
        self.queues["out"] = kwargs["out_queue"]

        self.queues["predict"] = queue_manager.get_queue("detect_predict", 8, multiprocessing_queue=False)
        self.queues["post"] = queue_manager.get_queue("detect_post", 8, multiprocessing_queue=False)
        self.queues["rotate"] = queue_manager.get_queue("detect_rotate", 8, multiprocessing_queue=False)
        self.create_threads()

    def create_threads(self):
        self._threads.append(
            FSThread(target=self.compile_thread, args=(self.queues["predict"],))
        )
        self._threads.append(
            FSThread(target=self.predict_thread, args=(self.queues["predict"], self.queues["post"]))
        )
        self._threads.append(
            FSThread(target=self.postprocess_thread, args=(self.queues["post"], self.queues["rotate"]))
        )
        for thread in self._threads:
            thread.start()
        logger.info("Started threads")

    def _resize_image(self, img):
        # TODO: overwrite in impl if required
        detect_image, scale, pads = self.compile_detection_image(
            img, is_square=True, pad_to=self.target
        )
        return (detect_image, scale, pads)

    def _rotate_image(self, img, angle):
        # TODO: overwrite in impl if required
        img, rotmat = self.rotate_image_by_angle(
            img, angle, *self.target
        )
        return (img, rotmat)

    def compile_batch(self, items, np_data):
        # TODO: overwrite in impl if required
        return items, np_data

    def compile_thread(self, out_queue):
        input_shutdown = False
        from time import time as now
        while True:
            logger.info("compile_thread waiting for item")
            stime = now()
            got_eof, in_batch = self.get_batch()
            stime = now() - stime
            logger.info("compile_thread got item in %.5f seconds", stime)
            batch = list()
            np_batch = list()
            for item in in_batch:
                detector_opts = item.setdefault("_detect_ops", {})
                if "scaled_img" not in detector_opts:
                    detect_image, scale, pads = self._resize_image(item["image"])
                    detector_opts["scale"] = scale
                    detector_opts["pads"] = pads
                    detector_opts["rotations"] = list(self.rotation)
                    detector_opts["rotmatrix"] = None  # the first "rotation" is always 0
                    img = detector_opts["scaled_img"] = detect_image
                else:
                    angle = detector_opts["rotations"][0]
                    img, rotmat = self._rotate_image(detector_opts["scaled_img"], angle)
                    detector_opts["rotmatrix"] = rotmat
                batch.append(item)
                np_batch.append(img)

            if batch:
                batch_data = np.array(np_batch, dtype="float32")
                batch, np_batch = self.compile_batch(batch, batch_data)
                out_queue.put((batch, np_batch))
                batch = []
                np_batch = []

            if got_eof:
                logger.debug("S3fd-amd main worker got EOF")
                out_queue.put("EOF")
                # Required to prevent hanging when less then BS items are in the
                # again queue and we won't receive new images.
                self.batch_size = 1
                if input_shutdown:
                    break
                input_shutdown = True

    def predict_batch(self, np_batch):
        raise NotImplemented()

    def predict_thread(self, in_queue, out_queue):
        input_shutdown = False
        from time import time as now
        while True:
            stime = now()
            batch = in_queue.get()
            stime = now() - stime
            logger.info("predict_thread got item in %.5f seconds", stime)
            if batch == "EOF":
                if input_shutdown:
                    break
                input_shutdown = True
                out_queue.put(batch)
                continue
            items, np_batch = batch
            predicted = self.predict_batch(np_batch)
            out_queue.put((items, predicted))

    def postprocess(self, items, predictions):
        raise NotImplemented()

    def postprocess_thread(self, in_queue, again_queue):
        # If -r is set we move images without found faces and remaining
        # rotations to a queue which is "merged" with the intial input queue.
        # This also means it is possible that we get data after an EOF.
        # This is handled by counting open rotation jobs and propagating
        # a second EOF as soon as we are really done through
        # the preprocsessing thread (detect_faces) and the prediction thread.
        open_rot_jobs = 0
        input_shutdown = False
        from time import time as now
        while True:
            stime = now()
            job = in_queue.get()
            stime = now() - stime
            logger.info("postprocess_thread got item in %.5f seconds", stime)
            if job == "EOF":
                logger.debug("S3fd-amd post processing got EOF")
                input_shutdown = True
            else:
                items, predictions = job
                predictions = self.postprocess(items, predictions)
                for prediction, item in zip(predictions, items):
                    detect_opts = item["_detect_ops"]
                    did_rotation = detect_opts["rotations"].pop(0) != 0
                    detected_faces = self.process_output(prediction, detect_opts)
                    if detected_faces:
                        item["detected_faces"] = detected_faces
                        del item["_detect_ops"]
                        self.finalize(item)
                        if did_rotation:
                            open_rot_jobs -= 1
                            logger.info("Found face after rotation.")
                    elif detect_opts["rotations"]:  # we have remaining rotations
                        logger.info("No face detected, remaining rotations: %s", detect_opts["rotations"])
                        if not did_rotation:
                            open_rot_jobs += 1
                        logger.info("Rotate face %s and try again.", item["filename"])
                        again_queue.put(item)
                    else:
                        logger.info("No face detected for %s.", item["filename"])
                        open_rot_jobs -= 1
                        item["detected_faces"] = []
                        del item["_detect_ops"]
                        self.finalize(item)
            if input_shutdown and open_rot_jobs <= 0:
                logger.debug("Sending second EOF")
                again_queue.put("EOF")
                self.finalize("EOF")
                break

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in rgb image
            Override for specific detector
            Must return a list of bounding box dicts (See module docstring)"""
            # TODO: switch this all so that extract pipeline only starts the threads
            # and calls raise_thread_exception on this class periodically.
            # This class.raise_thread_exception the aggregates all exceptions
            # and shuts everything down if any exception is raised...
        try:
            if not self.init:
                self.initialize(*args, **kwargs)
#             while not self.got_input_eof: # TODO:
#                 self.detect_and_raise_errors()
#                 import time
#                 time.sleep(0.1)
#             for thread in self._threads:
#                 thread.join()
        except ValueError as err:
            logger.error(err)
            exit(1)
        logger.debug("Detecting Faces (args: %s, kwargs: %s)", args, kwargs)


    def process_output(self, faces, opts):
        """ Compile found faces for output """
        logger.trace(
            "Processing Output: (faces: %s, rotation_matrix: %s)",
            faces, opts["rotmatrix"]
        )
        detected = []
        scale = opts["scale"]
        pad_l, pad_t = opts["pads"]
        rot = opts["rotmatrix"]
        for face in faces:
            face = self.to_bounding_box_dict(face[0], face[1], face[2], face[3])
            if isinstance(rot, np.ndarray):
                face = self.rotate_rect(face, rot)
            face = self.to_bounding_box_dict(
                (face["left"] - pad_l) / scale,
                (face["top"] - pad_t) / scale,
                (face["right"] - pad_l) / scale,
                (face["bottom"] - pad_t) / scale
            )
            detected.append(face)
        logger.trace("Processed Output: %s", detected)
        return detected


    # <<< GET MODEL >>> #
    @staticmethod
    def get_model(git_model_id, model_filename):
        """ Check if model is available, if not, download and unzip it """
        if model_filename is None:
            logger.debug("No model_filename specified. Returning None")
            return None
        if git_model_id is None:
            logger.debug("No git_model_id specified. Returning None")
            return None
        cache_path = os.path.join(os.path.dirname(__file__), ".cache")
        model = GetModel(model_filename, cache_path, git_model_id)
        return model.model_path

    # <<< DETECTION WRAPPER >>> #
    def run(self, *args, **kwargs):
        """ Parent detect process.
            This should always be called as the entry point so exceptions
            are passed back to parent.
            Do not override """
        try:
            logger.debug("Executing detector run function")
            self.detect_faces(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Caught exception in child process: %s: %s", os.getpid(), str(err))
            # Display traceback if in initialization stage
            if not self.init.is_set():
                logger.exception("Traceback:")
            tb_buffer = StringIO()
            traceback.print_exc(file=tb_buffer)
            logger.trace(tb_buffer.getvalue())
            exception = {"exception": (os.getpid(), tb_buffer)}
            self.queues["out"].put(exception)
            exit(1)

    # <<< FINALIZE METHODS>>> #
    def finalize(self, output):
        """ This should be called as the final task of each plugin
            Performs fianl processing and puts to the out queue """
        if isinstance(output, dict):
            logger.trace("Item out: %s", {key: val
                                          for key, val in output.items()
                                          if key != "image"})
            # Prevent zero size faces
            iheight, iwidth = output["image"].shape[:2]
            output["detected_faces"] = [
                f for f in output.get("detected_faces", list())
                if f["right"] > 0 and f["left"] < iwidth
                and f["bottom"] > 0 and f["top"] < iheight
            ]
            if self.min_size > 0 and output.get("detected_faces", None):
                output["detected_faces"] = self.filter_small_faces(output["detected_faces"])
        else:
            logger.trace("Item out: %s", output)
        self.queues["out"].put(output)

    def filter_small_faces(self, detected_faces):
        """ Filter out any faces smaller than the min size threshold """
        retval = list()
        for face in detected_faces:
            width = face["right"] - face["left"]
            height = face["bottom"] - face["top"]
            face_size = (width ** 2 + height ** 2) ** 0.5
            if face_size < self.min_size:
                logger.debug("Removing detected face: (face_size: %s, min_size: %s",
                             face_size, self.min_size)
                continue
            retval.append(face)
        return retval

    # <<< DETECTION IMAGE COMPILATION METHODS >>> #
    def compile_detection_image(self, input_image,  # pylint:disable=too-many-arguments
                                is_square=False, scale_up=False, to_rgb=False,
                                to_grayscale=False, pad_to=None):
        """ Compile the detection image """
        image = input_image.copy()
        if to_rgb:
            image = image[:, :, ::-1]
        elif to_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
        scale = self.set_scale(image, is_square=is_square, scale_up=scale_up)
        image = self.scale_image(image, scale, pad_to)
        if pad_to is None:
            return [image, scale]
        pad_left = int(pad_to[0] - int(input_image.shape[1] * scale)) // 2
        pad_top = int(pad_to[1] - int(input_image.shape[0] * scale)) // 2
        return [image, scale, (pad_left, pad_top)]

    def set_scale(self, image, is_square=False, scale_up=False):
        """ Set the scale factor for incoming image """
        height, width = image.shape[:2]
        if is_square:
            if isinstance(self.target, int):
                dims = (self.target ** 0.5, self.target ** 0.5)
                self.target = dims
            source = max(height, width)
            target = max(self.target)
        else:
            source = (width * height) ** 0.5
            if isinstance(self.target, tuple):
                #self.target = self.target[0] * self.target[1]
                target = self.target[0] * self.target[1]
            target = self.target ** 0.5

        if scale_up or target < source:
            scale = target / source
        else:
            scale = 1.0
        logger.trace("Detector scale: %s", scale)

        return scale

    @staticmethod
    def scale_image(image, scale, pad_to=None):
        """ Scale the image and optional pad to given size """
        # pylint: disable=no-member
        height, width = image.shape[:2]
        interpln = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
        if scale != 1.0:
            dims = (int(width * scale), int(height * scale))
            if scale < 1.0:
                logger.debug("Resizing image from %sx%s to %s. Scale=%s",
                             width, height, "x".join(str(i) for i in dims), scale)
            image = cv2.resize(image, dims, interpolation=interpln)
        if pad_to:
            image = Detector.pad_image(image, pad_to)
        return image

    @staticmethod
    def pad_image(image, target):
        """ Pad an image to a square """
        height, width = image.shape[:2]
        if width < target[0] or height < target[1]:
            pad_l = (target[0] - width) // 2
            pad_r = (target[0] - width) - pad_l
            pad_t = (target[1] - height) // 2
            pad_b = (target[1] - height) - pad_t
            img = cv2.copyMakeBorder(  # pylint:disable=no-member
                image, pad_t, pad_b, pad_l, pad_r,
                cv2.BORDER_CONSTANT, (0, 0, 0)  # pylint:disable=no-member
            )
            return img
        return image

    # <<< IMAGE ROTATION METHODS >>> #
    @staticmethod
    def get_rotation_angles(rotation):
        """ Set the rotation angles. Includes backwards compatibility for the
            'on' and 'off' options:
                - 'on' - increment 90 degrees
                - 'off' - disable
                - 0 is prepended to the list, as whatever happens, we want to
                  scan the image in it's upright state """
        rotation_angles = [0]

        if not rotation or rotation.lower() == "off":
            logger.debug("Not setting rotation angles")
            return rotation_angles

        if rotation.lower() == "on":
            rotation_angles.extend(range(90, 360, 90))
        else:
            passed_angles = [
                int(angle)
                for angle in rotation.split(",")
                if int(angle) != 0
            ]
            if len(passed_angles) == 1:
                rotation_step_size = passed_angles[0]
                rotation_angles.extend(range(rotation_step_size,
                                             360,
                                             rotation_step_size))
            elif len(passed_angles) > 1:
                rotation_angles.extend(passed_angles)

        logger.debug("Rotation Angles: %s", rotation_angles)
        return rotation_angles

    def rotate_image(self, image, angle):
        """ Rotate the image by given angle and return
            Image with rotation matrix """
        if angle == 0:
            return image, None
        return self.rotate_image_by_angle(image, angle)

    @staticmethod
    def rotate_rect(bounding_box, rotation_matrix):
        """ Rotate a bounding box dict based on the rotation_matrix"""
        logger.trace("Rotating bounding box")
        bounding_box = rotate_landmarks(bounding_box, rotation_matrix)
        return bounding_box

    @staticmethod
    def rotate_image_by_angle(image, angle,
                              rotated_width=None, rotated_height=None):
        """ Rotate an image by a given angle.
            From: https://stackoverflow.com/questions/22041699 """

        logger.trace("Rotating image: (angle: %s, rotated_width: %s, rotated_height: %s)",
                     angle, rotated_width, rotated_height)
        height, width = image.shape[:2]
        image_center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(  # pylint: disable=no-member
            image_center, -1.*angle, 1.)
        if rotated_width is None or rotated_height is None:
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            if rotated_width is None:
                rotated_width = int(height*abs_sin + width*abs_cos)
            if rotated_height is None:
                rotated_height = int(height*abs_cos + width*abs_sin)
        rotation_matrix[0, 2] += rotated_width/2 - image_center[0]
        rotation_matrix[1, 2] += rotated_height/2 - image_center[1]
        logger.trace("Rotated image: (rotation_matrix: %s", rotation_matrix)
        return (cv2.warpAffine(image,  # pylint: disable=no-member
                               rotation_matrix,
                               (rotated_width, rotated_height)),
                rotation_matrix)

    # << QUEUE METHODS >> #
    def get_item(self):
        """
        Yield one item from the input or rotation
        queue while prioritizing rotation queue to
        prevent deadlocks.
        """
        try:
            item = self.queues["rotate"].get(block=self.got_input_eof)
            logger.info("Got item from again queue")
            return item
        except queue.Empty:
            pass
        item = self.queues["in"].get()
        if isinstance(item, dict):
            logger.trace("Item in: %s", item["filename"])
        else:
            logger.trace("Item in: %s", item)
        if item == "EOF":
            self.got_input_eof = True
            logger.debug("In Queue Exhausted")
            # Re-put EOF into queue for other threads
            self.queues["in"].put(item)
        return item

    def get_batch(self):
        """ Get items from the queue in batches of
            self.batch_size

            First item in output tuple indicates whether the
            queue is exhausted.
            Second item is the batch

            Remember to put "EOF" to the out queue after processing
            the final batch """
        exhausted = False
        batch = list()
        for _ in range(self.batch_size):
            item = self.get_item()
            if item == "EOF":
                exhausted = True
                break
            batch.append(item)
        logger.trace("Returning batch size: %s", len(batch))
        return (exhausted, batch)

    # <<< MISC METHODS >>> #
    def get_vram_free(self):
        """ Return free and total VRAM on card with most VRAM free"""
        stats = GPUStats()
        vram = stats.get_card_most_free(supports_plaidml=self.supports_plaidml)
        logger.verbose("Using device %s with %sMB free of %sMB",
                       vram["device"],
                       int(vram["free"]),
                       int(vram["total"]))
        return int(vram["card_id"]), int(vram["free"]), int(vram["total"])

    @staticmethod
    def to_bounding_box_dict(left, top, right, bottom):
        """ Return a dict for the bounding box """
        return dict(left=int(round(left)),
                    right=int(round(right)),
                    top=int(round(top)),
                    bottom=int(round(bottom)))

    def set_predetected(self, width, height):
        """ Set a bounding box dict for predetected faces """
        # Predetected_face is used for sort tool.
        # Landmarks should not be extracted again from predetected faces,
        # because face data is lost, resulting in a large variance
        # against extract from original image
        logger.debug("Setting predetected face")
        return [self.to_bounding_box_dict(0, 0, width, height)]
