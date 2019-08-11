#!/usr/bin/env python3
""" Converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging

import cv2
import numpy as np

from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
from lib.MyTimeit import TimeIt
#timeit = None

class Converter():
    """ Swap a source face with a target """
    def __init__(self, output_dir, output_size, output_has_mask,
                 draw_transparent, pre_encode, arguments, configfile=None):
        logger.debug("Initializing %s: (output_dir: '%s', output_size: %s,  output_has_mask: %s, "
                     "draw_transparent: %s, pre_encode: %s, arguments: %s, configfile: %s)",
                     self.__class__.__name__, output_dir, output_size, output_has_mask,
                     draw_transparent, pre_encode, arguments, configfile)
        logger.info("Set up timeit")
        self.output_dir = output_dir
        self.draw_transparent = draw_transparent
        self.writer_pre_encode = pre_encode
        self.scale = arguments.output_scale / 100
        self.output_size = output_size
        self.output_has_mask = output_has_mask
        self.args = arguments
        self.configfile = configfile
        self.adjustments = dict(box=None, mask=None, color=None, seamless=None, scaling=None)
        self.load_plugins()
        logger.debug("Initialized %s", self.__class__.__name__)

    def reinitialize(self, config):
        """ reinitialize converter """
        logger.debug("Reinitializing converter")
        self.adjustments = dict(box=None, mask=None, color=None, seamless=None, scaling=None)
        self.load_plugins(config=config, disable_logging=True)
        logger.debug("Reinitialized converter")

    def load_plugins(self, config=None, disable_logging=False):
        """ Load the requested adjustment plugins """
        logger.debug("Loading plugins. config: %s", config)
        self.adjustments["box"] = PluginLoader.get_converter(
            "mask",
            "box_blend",
            disable_logging=disable_logging)("none",
                                             self.output_size,
                                             configfile=self.configfile,
                                             config=config)

        self.adjustments["mask"] = PluginLoader.get_converter(
            "mask",
            "mask_blend",
            disable_logging=disable_logging)(self.args.mask_type,
                                             self.output_size,
                                             self.output_has_mask,
                                             configfile=self.configfile,
                                             config=config)

        if self.args.color_adjustment != "none" and self.args.color_adjustment is not None:
            self.adjustments["color"] = PluginLoader.get_converter(
                "color",
                self.args.color_adjustment,
                disable_logging=disable_logging)(configfile=self.configfile, config=config)

        if self.args.scaling != "none" and self.args.scaling is not None:
            self.adjustments["scaling"] = PluginLoader.get_converter(
                "scaling",
                self.args.scaling,
                disable_logging=disable_logging)(configfile=self.configfile, config=config)
        logger.debug("Loaded plugins: %s", self.adjustments)

    def process(self, in_queue, out_queue, completion_queue=None):
        """ Process items from the queue """
        logger.debug("Starting convert process. (in_queue: %s, out_queue: %s, completion_queue: "
                     "%s)", in_queue, out_queue, completion_queue)
        self.timeit = TimeIt()
        count = 0
        while True:
            with self.timeit.log("Converter.in_queue.get"):
                item = in_queue.get()
            if item == "EOF":
                logger.debug("EOF Received")
                logger.debug("Patch queue finished")
                # Signal EOF to other processes in pool
                logger.debug("Putting EOF back to in_queue")
                in_queue.put(item)
                break
            logger.trace("Patch queue got: '%s'", item["filename"])

            try:
                with self.timeit.log("Converter.patch_image"):
                    image = self.patch_image(item)
            except Exception as err:  # pylint: disable=broad-except
                # Log error and output original frame
                logger.error("Failed to convert image: '%s'. Reason: %s",
                             item["filename"], str(err))
                image = item["image"]
                # UNCOMMENT THIS CODE BLOCK TO PRINT TRACEBACK ERRORS
                # import sys
                # import traceback
                # exc_info = sys.exc_info()
                # traceback.print_exception(*exc_info)

            count += 1
            if count % (10*16) == 0:
                self.timeit.print_summary()

            logger.trace("Out queue put: %s", item["filename"])
            with self.timeit.log("Converter.out_queue.put"):
                out_queue.put((item["filename"], image))
        logger.debug("Completed convert process")
        # Signal that this process has finished
        if completion_queue is not None:
            completion_queue.put(1)

    def patch_image(self, predicted):
        """ Patch the image """
        logger.trace("Patching image: '%s'", predicted["filename"])
        frame_size = (predicted["image"].shape[1], predicted["image"].shape[0])
        with self.timeit.log("Converter.get_new_image"):
            new_image = self.get_new_image(predicted, frame_size)
        with self.timeit.log("Converter.post_warp_adjustments"):
            patched_face = self.post_warp_adjustments(predicted, new_image)
        with self.timeit.log("Converter.scale_image"):
            patched_face = self.scale_image(patched_face)
        patched_face = np.rint(patched_face * 255.0).astype("uint8")
        if self.writer_pre_encode is not None:
            with self.timeit.log("Converter.writer_pre_encode"):
                patched_face = self.writer_pre_encode(patched_face)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

# original
#    ALL          PER CALL     COUNT   
#   886.845171     0.369519     2400  Converter.patch_image
#   414.987130     0.172911     2400    Converter.post_warp_adjustments
#   328.976959     0.137074     2400    Converter.get_new_image
#   205.606595     0.085669     2400      Converter.concatenate
#    49.260268     0.020525     2400      Converter image / 255
#    43.625085     0.018177     2400      Converter.clip
#    16.563870     0.007616     2175      Converter.pre_warp_adjustments
#    12.432135     0.005716     2175      Converter.cv2.warpAffine
#     0.017605     0.000008     2175      Converter.detected_face.reference_face
#     0.036682     0.000015     2400    Converter.scale_image
#    76.433431     0.031861     2399  Converter.out_queue.put
#    42.185114     0.017577     2400  Converter.in_queue.get

# earlier cast to float
# Thread: MainThread (140715490702976)
#    ALL          PER CALL     COUNT   
#   330.911167     0.344699      960  Converter.patch_image
#   170.045525     0.177131      960    Converter.post_warp_adjustments
#    95.983902     0.099983      960    Converter.get_new_image
#    36.382884     0.037899      960      Converter.get_new_image.concat
#    32.896013     0.034267      960      Converter image / 255
#    11.434691     0.011911      960      Converter.clip
#     7.988529     0.008856      902      Converter.pre_warp_adjustments
#     5.156356     0.005717      902      Converter.cv2.warpAffine
#     1.324564     0.001380      960      Converter.get_new_image.zeros
#     0.008408     0.000009      960    Converter.scale_image
#    42.458343     0.044274      959  Converter.out_queue.put
#    25.423412     0.026483      960  Converter.in_queue.get

# concat replaced
#   152.674031     0.318071      480  Converter.patch_image
#    81.286563     0.169347      480    Converter.post_warp_adjustments
#    40.332066     0.084025      480    Converter.get_new_image
#    15.046225     0.031346      480      Converter.get_new_image.concat
#    13.517876     0.028162      480      Converter image / 255
#     5.539292     0.011540      480      Converter.clip
#     3.518051     0.008087      435      Converter.pre_warp_adjustments
#     2.412512     0.005546      435      Converter.cv2.warpAffine
#     0.003057     0.000006      480    Converter.scale_image
#    22.317739     0.046592      479  Converter.out_queue.put
#    17.059331     0.035540      480  Converter.in_queue.get

# div moved to  after "concated"
#    ALL          PER CALL     COUNT   
#   340.566147     0.304077     1120  Converter.patch_image
#   192.698753     0.172052     1120    Converter.post_warp_adjustments
#    79.974632     0.071406     1120    Converter.get_new_image
#    41.820742     0.037340     1120      Converter.get_new_image.concat
#    22.734484     0.020299     1120      Converter.clip
#     8.573580     0.008340     1028      Converter.pre_warp_adjustments
#     6.205790     0.006037     1028      Converter.cv2.warpAffine
#     0.006895     0.000006     1120    Converter.scale_image
#    56.843160     0.050798     1119  Converter.out_queue.put
#    28.520413     0.025465     1120  Converter.in_queue.get

# clip uses out parameter
# Thread: MainThread (139787813049984)
#    ALL          PER CALL     COUNT   
#    99.277319     0.310242      320  Converter.patch_image
#    59.783738     0.186824      320    Converter.post_warp_adjustments
#    19.986927     0.062459      320    Converter.get_new_image
#    12.454857     0.038921      320      Converter.get_new_image.concat
#     3.430781     0.010721      320      Converter.clip
#     2.427035     0.008826      275      Converter.pre_warp_adjustments
#     1.505444     0.005474      275      Converter.cv2.warpAffine
#     0.001786     0.000006      320    Converter.scale_image
#    15.198569     0.047496      320  Converter.in_queue.get
#    14.540891     0.045583      319  Converter.out_queue.put


    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply box manipulations """
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))

        with self.timeit.log("Converter.get_new_image.concat"):
            placeholder = np.zeros((frame_size[1], frame_size[0], 4), dtype="float32")
            placeholder[:, :, :3] = predicted["image"]  # faster as concat
            placeholder /= 255.0  # faster to include the zero data instead to slice or do it before (which would include casting)

        for new_face, detected_face in zip(predicted["swapped_faces"],
                                           predicted["detected_faces"]):
            predicted_mask = new_face[:, :, -1] if new_face.shape[2] == 4 else None
            new_face = new_face[:, :, :3]
            src_face = detected_face.reference_face
            interpolator = detected_face.reference_interpolators[1]

            with self.timeit.log("Converter.pre_warp_adjustments"):
                new_face = self.pre_warp_adjustments(src_face, new_face, detected_face, predicted_mask)

            with self.timeit.log("Converter.cv2.warpAffine"):
                # Warp face with the mask
                placeholder = cv2.warpAffine(  # pylint: disable=no-member
                    new_face,
                    detected_face.reference_matrix,
                    frame_size,
                    placeholder,
                    flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                    borderMode=cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member

        with self.timeit.log("Converter.clip"):
            np.clip(placeholder, 0.0, 1.0, out=placeholder)
        logger.trace("Got filename: '%s'. (placeholders: %s)",
                     predicted["filename"], placeholder.shape)

        return placeholder

    def pre_warp_adjustments(self, old_face, new_face, detected_face, predicted_mask):
        """ Run the pre-warp adjustments """
        logger.trace("old_face shape: %s, new_face shape: %s, predicted_mask shape: %s",
                     old_face.shape, new_face.shape,
                     predicted_mask.shape if predicted_mask is not None else None)
        new_face = self.adjustments["box"].run(new_face)
        new_face, raw_mask = self.get_image_mask(new_face, detected_face, predicted_mask)
        if self.adjustments["color"] is not None:
            new_face = self.adjustments["color"].run(old_face, new_face, raw_mask)
        if self.adjustments["seamless"] is not None:
            new_face = self.adjustments["seamless"].run(old_face, new_face, raw_mask)
        logger.trace("returning: new_face shape %s", new_face.shape)
        return new_face

    def get_image_mask(self, new_face, detected_face, predicted_mask):
        """ Get the image mask """
        logger.trace("Getting mask. Image shape: %s", new_face.shape)
        mask, raw_mask = self.adjustments["mask"].run(detected_face, predicted_mask)
        if new_face.shape[2] == 4:
            logger.trace("Combining mask with alpha channel box mask")
            new_face[:, :, -1] = np.minimum(new_face[:, :, -1], mask.squeeze())
        else:
            logger.trace("Adding mask to alpha channel")
            new_face = np.concatenate((new_face, mask), -1)
        new_face = np.clip(new_face, 0.0, 1.0)
        logger.trace("Got mask. Image shape: %s", new_face.shape)
        return new_face, raw_mask



# initial
# Thread: MainThread (139921978742400)
#    ALL          PER CALL     COUNT   
#   199.021895     0.310972      640  Converter.patch_image
#   119.703733     0.187037      640    Converter.post_warp_adjustments
#    51.355958     0.080244      640      Converter.post_warp_adjustments.background calc
#    26.774223     0.041835      640      Converter.post_warp_adjustments.np.repeat
#    17.383283     0.027161      640      Converter.post_warp_adjustments.apply mask add
#    17.380628     0.027157      640      Converter.post_warp_adjustments.apply mask mul
#    39.735953     0.062087      640    Converter.get_new_image
#    25.075697     0.039181      640      Converter.get_new_image.concat
#     6.087227     0.009511      640      Converter.clip
#     4.962581     0.008355      594      Converter.pre_warp_adjustments
#     3.263558     0.005494      594      Converter.cv2.warpAffine
#     0.003436     0.000005      640    Converter.scale_image
#    31.977217     0.050043      639  Converter.out_queue.put
#    19.660525     0.030720      640  Converter.in_queue.get

# mask fixes
#    ALL          PER CALL     COUNT   
#   592.579174     0.284894     2080  Converter.patch_image
#   323.014564     0.155295     2080    Converter.post_warp_adjustments
#   148.186818     0.071244     2080      Converter.post_warp_adjustments.background calc
#    86.587287     0.041629     2080      Converter.post_warp_adjustments.apply mask mul
#    62.717508     0.030153     2080      Converter.post_warp_adjustments.apply mask add
#   126.571387     0.060852     2080    Converter.get_new_image
#    73.443612     0.035309     2080      Converter.get_new_image.concat
#    22.260009     0.010702     2080      Converter.clip
#    17.649798     0.008996     1962      Converter.pre_warp_adjustments
#    11.767189     0.005998     1962      Converter.cv2.warpAffine
#     0.015159     0.000007     2080    Converter.scale_image
#   147.994378     0.071185     2079  Converter.out_queue.put
#    40.618026     0.019528     2080  Converter.in_queue.get




# Thread: MainThread (140154532656768)
#    ALL          PER CALL     COUNT   
#   203.267760     0.317606      640  Converter.patch_image
#   129.681146     0.202627      640    Converter.post_warp_adjustments
#    40.475694     0.063243      640      Converter.post_warp_adjustments.background calc
#    24.583730     0.038412      640      Converter.post_warp_adjustments.apply mask mul
#    22.846998     0.035698      640      Converter.post_warp_adjustments.apply mask add
#    33.667381     0.052605      640    Converter.get_new_image
#    19.272453     0.030113      640      Converter.get_new_image.concat
#     6.204458     0.009694      640      Converter.clip
#     4.559653     0.007650      596      Converter.pre_warp_adjustments
#     3.236168     0.005430      596      Converter.cv2.warpAffine
#     0.003294     0.000005      640    Converter.scale_image
#    24.454575     0.038270      639  Converter.out_queue.put
#    18.446828     0.028823      640  Converter.in_queue.get




#    ALL          PER CALL     COUNT   
#   432.763537     0.270477     1600  Converter.patch_image
#   236.658927     0.147912     1600    Converter.post_warp_adjustments
#   109.274762     0.068297     1600      Converter.post_warp_adjustments.background calc
#    63.298678     0.039562     1600      Converter.post_warp_adjustments.apply mask mul
#    44.970891     0.028107     1600      Converter.post_warp_adjustments.apply mask add
#    92.010180     0.057506     1600    Converter.get_new_image
#    53.150059     0.033219     1600      Converter.get_new_image.concat
#    16.151920     0.010095     1600      Converter.clip
#    12.926733     0.008595     1504      Converter.pre_warp_adjustments
#     8.759335     0.005824     1504      Converter.cv2.warpAffine
#     0.009251     0.000006     1600    Converter.scale_image
#   111.570375     0.069775     1599  Converter.out_queue.put
#    33.054469     0.020659     1600  Converter.in_queue.get                              



#   172.070568     0.215088      800  Converter.patch_image
#    94.553206     0.118192      800    Converter.post_warp_adjustments
#    36.847095     0.046059      800      Converter.post_warp_adjustments.background calc
#    31.113633     0.038892      800      Converter.post_warp_adjustments.apply mask mul
#    20.176571     0.025221      800      Converter.post_warp_adjustments.apply mask add
#    44.660558     0.055826      800    Converter.get_new_image
#    26.127226     0.032659      800      Converter.get_new_image.concat
#     8.068309     0.010085      800      Converter.clip
#     6.096823     0.008217      742      Converter.pre_warp_adjustments
#     3.833489     0.005166      742      Converter.cv2.warpAffine
#     0.004364     0.000005      800    Converter.scale_image
#    63.123336     0.079003      799  Converter.out_queue.put
#    22.584089     0.028230      800  Converter.in_queue.get
                              

#    ALL          PER CALL     COUNT   
#   265.186240     0.207177     1280  Converter.patch_image
#   133.704767     0.104457     1280    Converter.post_warp_adjustments
#    53.033429     0.041432     1280      Converter.post_warp_adjustments.background calc
#    49.159798     0.038406     1280      Converter.post_warp_adjustments.apply mask mul
#    21.566511     0.016849     1280      Converter.post_warp_adjustments.apply mask add
#    70.912294     0.055400     1280    Converter.get_new_image
#    41.677989     0.032561     1280      Converter.get_new_image.concat
#    12.514353     0.009777     1280      Converter.clip
#     9.542924     0.008019     1190      Converter.pre_warp_adjustments
#     6.512927     0.005473     1190      Converter.cv2.warpAffine
#     0.006713     0.000005     1280    Converter.scale_image
#   103.026066     0.080552     1279  Converter.out_queue.put
#    27.707409     0.021646     1280  Converter.in_queue.get




    def post_warp_adjustments(self, predicted, new_image):
        """ Apply fixes to the image after warping """
        if self.adjustments["scaling"] is not None:
            with self.timeit.log("Converter.post_warp_adjustments.scale"):
                new_image = self.adjustments["scaling"].run(new_image)

        if self.draw_transparent:
            frame = new_image
        else:
            mask = new_image[:, :, -1:]
            foreground = np.array(new_image[:, :, :3], dtype="float32")
            with self.timeit.log("Converter.post_warp_adjustments.background calc"):
                background = np.array(predicted["image"][:, :, :3], dtype="float32")
                background /= 255.0
                background *= (1.0 - mask)
            with self.timeit.log("Converter.post_warp_adjustments.apply mask mul"):
                foreground *= mask
            with self.timeit.log("Converter.post_warp_adjustments.apply mask add"):
                background += foreground
                frame = background

        np.clip(frame, 0.0, 1.0, out=frame)
        return frame

    def scale_image(self, frame):
        """ Scale the image if requested """
        if self.scale == 1:
            return frame
        logger.trace("source frame: %s", frame.shape)
        interp = cv2.INTER_CUBIC if self.scale > 1 else cv2.INTER_AREA  # pylint: disable=no-member
        dims = (round((frame.shape[1] / 2 * self.scale) * 2),
                round((frame.shape[0] / 2 * self.scale) * 2))
        frame = cv2.resize(frame, dims, interpolation=interp)  # pylint: disable=no-member
        logger.trace("resized frame: %s", frame.shape)
        return np.clip(frame, 0.0, 1.0)
