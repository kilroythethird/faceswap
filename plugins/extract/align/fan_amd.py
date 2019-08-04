#!/usr/bin/env python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""
import cv2
import numpy as np
import keras
from keras import backend as K
from lib.multithreading import FSThread
from lib.queue_manager import queue_manager
from ._base import Aligner, logger


class Align(Aligner):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = 9
        model_filename = "face-alignment-network_2d4_keras_v1.h5"
        super().__init__(git_model_id=git_model_id,
                         model_filename=model_filename,
                         colorspace="RGB",
                         input_size=256,
                         **kwargs)
        self.vram = 2240
        self.model = None
        self.reference_scale = 195
        self.supports_plaidml = True

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing Face Alignment Network (AMD)...")
            logger.debug("fan initialize: (args: %s kwargs: %s)", args, kwargs)
            self.model = FAN(self.model_path)
            self.batch_size = self.config["batch-size"]
            logger.verbose("Starting FAN-AMD with batchsize of %i.", self.batch_size)
            self.init.set()
            logger.info("Initialized Face Alignment Network (AMD).")
        except Exception as err:
            self.error.set()
            raise err

    def prediction_thread(self, in_queue, out_queue):
        while True:
            items = in_queue.get()
            if items == "EOF":
                logger.debug("FAN prediction_thread got EOF")
                out_queue.put(items)
                break
            infos, images, items = items
            predictions = self.model.predict(images)
            out_queue.put((predictions, infos, items))

    def post_processing_thread(self, in_queue):
        while True:
            job = in_queue.get()
            if job == "EOF":
                logger.debug("FAN post_processing_thread worker got EOF")
                self.finalize(job)
                break
            for pred, infos, item in zip(*job):
                pts_img = self.get_pts_from_predict(pred, infos["center"], infos["scale"])
                pts_img = [(int(pt[0]), int(pt[1])) for pt in pts_img]
                item["landmarks"].append(pts_img)
                if len(item["landmarks"]) == len(item["detected_faces"]):
                    self.finalize(item)

    def align(self, *args, **kwargs):
        """ Process landmarks """
        if not self.init:
            self.initialize(*args, **kwargs)
        logger.debug("Launching Align: (args: %s kwargs: %s)", args, kwargs)
        queue_size = 2
        queue_manager.add_queue("fan_work", queue_size, False)
        queue_manager.add_queue("fan_done", queue_size, False)
        work_queue = queue_manager.get_queue("fan_work", queue_size)
        done_queue = queue_manager.get_queue("fan_done", queue_size)
        worker = FSThread(target=self.prediction_thread, args=(work_queue, done_queue))
        worker.start()
        post_worker = FSThread(target=self.post_processing_thread, args=(done_queue,))
        post_worker.start()
        got_eof = False
        batches = list()
        for item in self.get_item():
            if item == "EOF":
                got_eof = True
            else:
                image = self.convert_color(item["image"])
                data = None
                try:
                    data = self.process_landmarks(image, item["detected_faces"])
                except ValueError as err:
                    logger.warning("Image '%s' could not be processed. This may be due"
                                   " to corrupted data: %s", item["filename"], str(err))
                item["landmarks"] = list()
                if not data and not got_eof:
                    item["detected_faces"] = list()
                    self.finalize(item)
                    continue
                batches.extend([(img, item) for img in data])

            while len(batches) >= self.batch_size or (batches and got_eof):
                batch = batches[:self.batch_size]
                batches = batches[len(batch):]
                infos = [x[0] for x in batch]
                images = np.array([x["image"] for x in infos]).astype("float32")
                items = [x[1] for x in batch]
                images = images.transpose((0, 3, 1, 2)) / 255.0
                work_queue.put((infos, images, items))

            if got_eof:
                work_queue.put("EOF")
                break

        worker.join()
        post_worker.join()
        logger.debug("Completed Align")

    def process_landmarks(self, image, detected_faces):
        """ Align image and process landmarks """
        retval = list()
        for detected_face in detected_faces:
            feed_dict = self.align_image(detected_face, image)
            self.normalize_face(feed_dict)
            retval.append(feed_dict)
        return retval

    # DETECTED FACE BOUNDING BOX PROCESSING
    def align_image(self, detected_face, image):
        """ Get center and scale, crop and align image around center """
        logger.trace("Aligning image around center")
        center, scale = self.get_center_scale(detected_face)
        image = self.crop(image, center, scale)
        logger.trace("Aligned image around center")
        return dict(image=image, center=center, scale=scale)

    def get_center_scale(self, detected_face):
        """ Get the center and set scale of bounding box """
        logger.trace("Calculating center and scale")
        center = np.array([(detected_face["left"] + detected_face["right"]) / 2.0,
                           (detected_face["top"] + detected_face["bottom"]) / 2.0])

        height = detected_face["bottom"] - detected_face["top"]
        width = detected_face["right"] - detected_face["left"]

        center[1] -= height * 0.12

        scale = (width + height) / self.reference_scale

        logger.trace("Calculated center and scale: %s, %s", center, scale)
        return center, scale

    def crop(self, image, center, scale):  # pylint:disable=too-many-locals
        """ Crop image around the center point """
        logger.trace("Cropping image")
        is_color = image.ndim > 2
        v_ul = self.transform([1, 1], center, scale, self.input_size).astype(np.int)
        v_br = self.transform([self.input_size, self.input_size],
                              center,
                              scale,
                              self.input_size).astype(np.int)
        if is_color:
            new_dim = np.array([v_br[1] - v_ul[1],
                                v_br[0] - v_ul[0],
                                image.shape[2]],
                               dtype=np.int32)
            new_img = np.zeros(new_dim, dtype=np.uint8)
        else:
            new_dim = np.array([v_br[1] - v_ul[1],
                                v_br[0] - v_ul[0]],
                               dtype=np.int)
            new_img = np.zeros(new_dim, dtype=np.uint8)
        height = image.shape[0]
        width = image.shape[1]
        new_x = np.array([max(1, -v_ul[0] + 1), min(v_br[0], width) - v_ul[0]],
                         dtype=np.int32)
        new_y = np.array([max(1, -v_ul[1] + 1),
                          min(v_br[1], height) - v_ul[1]],
                         dtype=np.int32)
        old_x = np.array([max(1, v_ul[0] + 1), min(v_br[0], width)],
                         dtype=np.int32)
        old_y = np.array([max(1, v_ul[1] + 1), min(v_br[1], height)],
                         dtype=np.int32)
        if is_color:
            new_img[new_y[0] - 1:new_y[1],
                    new_x[0] - 1:new_x[1]] = image[old_y[0] - 1:old_y[1],
                                                   old_x[0] - 1:old_x[1], :]
        else:
            new_img[new_y[0] - 1:new_y[1],
                    new_x[0] - 1:new_x[1]] = image[old_y[0] - 1:old_y[1],
                                                   old_x[0] - 1:old_x[1]]

        if new_img.shape[0] < self.input_size:
            interpolation = cv2.INTER_CUBIC  # pylint:disable=no-member
        else:
            interpolation = cv2.INTER_AREA  # pylint:disable=no-member

        new_img = cv2.resize(new_img,  # pylint:disable=no-member
                             dsize=(int(self.input_size), int(self.input_size)),
                             interpolation=interpolation)
        logger.trace("Cropped image")
        return new_img

    @staticmethod
    def transform(point, center, scale, resolution):
        """ Transform Image """
        logger.trace("Transforming Points")
        pnt = np.array([point[0], point[1], 1.0])
        hscl = 200.0 * scale
        eye = np.eye(3)
        eye[0, 0] = resolution / hscl
        eye[1, 1] = resolution / hscl
        eye[0, 2] = resolution * (-center[0] / hscl + 0.5)
        eye[1, 2] = resolution * (-center[1] / hscl + 0.5)
        eye = np.linalg.inv(eye)
        retval = np.matmul(eye, pnt)[0:2]
        logger.trace("Transformed Points: %s", retval)
        return retval

    def get_pts_from_predict(self, prediction, center, scale):
        """ Get points from predictor """
        logger.trace("Obtain points from prediction")
        var_b = prediction.reshape((prediction.shape[0],
                                    prediction.shape[1] * prediction.shape[2]))
        var_c = var_b.argmax(1).reshape((prediction.shape[0],
                                         1)).repeat(2,
                                                    axis=1).astype(np.float)
        var_c[:, 0] %= prediction.shape[2]
        var_c[:, 1] = np.apply_along_axis(
            lambda x: np.floor(x / prediction.shape[2]),
            0,
            var_c[:, 1])

        for i in range(prediction.shape[0]):
            pt_x, pt_y = int(var_c[i, 0]), int(var_c[i, 1])
            if pt_x > 0 and pt_x < 63 and pt_y > 0 and pt_y < 63:
                diff = np.array([prediction[i, pt_y, pt_x+1]
                                 - prediction[i, pt_y, pt_x-1],
                                 prediction[i, pt_y+1, pt_x]
                                 - prediction[i, pt_y-1, pt_x]])

                var_c[i] += np.sign(diff)*0.25

        var_c += 0.5
        retval = [self.transform(var_c[i], center, scale, prediction.shape[2])
                  for i in range(prediction.shape[0])]
        logger.trace("Obtained points from prediction: %s", retval)

        return retval


class TorchBatchNorm2D(keras.engine.base_layer.Layer):
    """" Required for FAN_keras model """
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, **kwargs):
        super(TorchBatchNorm2D, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self._epsilon_const = K.constant(self.epsilon, dtype='float32')

        self.built = False
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_variance = None

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError("Axis {} of input tensor should have a "
                             "defined dimension but the layer received "
                             "an input with  shape {}."
                             .format(str(self.axis), str(input_shape)))
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer='ones',
                                     regularizer=None,
                                     constraint=None)
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer='zeros',
                                    regularizer=None,
                                    constraint=None)
        self.moving_mean = self.add_weight(shape=shape,
                                           name='moving_mean',
                                           initializer='zeros',
                                           trainable=False)
        self.moving_variance = self.add_weight(shape=shape,
                                               name='moving_variance',
                                               initializer='ones',
                                               trainable=False)
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        broadcast_moving_mean = K.reshape(self.moving_mean, broadcast_shape)
        broadcast_moving_variance = K.reshape(self.moving_variance,
                                              broadcast_shape)
        broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
        broadcast_beta = K.reshape(self.beta, broadcast_shape)
        invstd = (
            K.ones(shape=broadcast_shape, dtype='float32')
            / K.sqrt(broadcast_moving_variance + self._epsilon_const)
        )

        return((inputs - broadcast_moving_mean)
               * invstd
               * broadcast_gamma
               + broadcast_beta)

    def get_config(self):
        config = {'axis': self.axis,
                  'momentum': self.momentum,
                  'epsilon': self.epsilon}
        base_config = super(TorchBatchNorm2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FAN():
    """
    Converted from pyTorch from
    https://github.com/1adrianb/face-alignment
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """ Load the Keras Model """
        logger.verbose("Initializing Face Alignment Network model (Keras version).")
        self.model = keras.models.load_model(
            self.model_path,
            custom_objects={'TorchBatchNorm2D': TorchBatchNorm2D}
        )

    def predict(self, feed_item):
        """ Predict landmarks in session """
        pred = self.model.predict(feed_item)
        return pred[-1].reshape((feed_item.shape[0], 68, 64, 64))
