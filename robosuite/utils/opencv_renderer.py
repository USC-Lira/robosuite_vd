"""
opencv renderer class.
"""
import cv2
import numpy as np


class OpenCVRenderer:
    def __init__(self, sim):
        # FIXME(dhanush): update this appropriately - need to get screen dimensions
        self.width = 1920
        self.height = 1080

        self.sim = sim
        self.camera_name = self.sim.model.camera_id2name(0)

        self.keypress_callback = None

    def set_camera(self, camera_id):
        """
        Set the camera view to the specified camera ID.
        Args:
            camera_id (int): id of the camera to set the current viewer to
        """
        self.camera_name = self.sim.model.camera_id2name(camera_id)

    # def render(self): #TODO: implement the dual rendering option
    #     # get frame with offscreen renderer (assumes that the renderer already exists)
    #     # im = self.sim.render(camera_name=self.camera_name, height=self.height, width=self.width)[..., ::-1]
    #     image = self.sim.render(camera_name=self.camera_name, height=self.height, width=self.width)
    #     im = image[..., ::-1]
    #     # write frame to window
    #     im = np.flip(im, axis=0)
    #     cv2.imshow("offscreen render", im)
    #     key = cv2.waitKey(1)
    #     if self.keypress_callback:
    #         self.keypress_callback(key)

    #     # import pdb; pdb.set_trace()

    #     assert image is not None, "Rendered image is None"
    #     print("I am giving image from OpenCV rendre")
    #     return image
        
    def render(self):
        # get frame with offscreen renderer (assumes that the renderer already exists)
        im = self.sim.render(camera_name=self.camera_name, height=self.height, width=self.width)[..., ::-1]

        # write frame to window
        im = np.flip(im, axis=0)
        cv2.imshow("offscreen render", im)
        key = cv2.waitKey(1)
        if self.keypress_callback:
            self.keypress_callback(key)

    def render_gaze(self, gaze_data):
        # get frame with offscreen renderer (assumes that the renderer already exists)
        im = self.sim.render(camera_name=self.camera_name, height=self.height, width=self.width)[..., ::-1]

        # write frame to window
        im = np.flip(im, axis=0)

        #Before displaying the image we draw the marker using the gaze information
        gaze_im = cv2.drawMarker(np.uint8(im.copy()), (int(gaze_data[0] * self.width), int(gaze_data[1] * self.height)), 
                                 color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=100, thickness=2)


        # import pdb; pdb.set_trace()

        cv2.imshow("offscreen render", gaze_im)
        key = cv2.waitKey(1)
        if self.keypress_callback:
            self.keypress_callback(key)



    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback

    def close(self):
        """
        Any cleanup to close renderer.
        """

        # NOTE: assume that @sim will get cleaned up outside the renderer - just delete the reference
        self.sim = None

        # close window
        cv2.destroyAllWindows()
