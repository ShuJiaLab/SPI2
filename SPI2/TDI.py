# TDI.py
from egrabber import * # You need to intall first: python -m pip install 'C:\Program Files\Euresys\eGrabber\python\egrabber-23.08.0.17-py2.py3-none-any.whl'
from ctypes import cast, POINTER, c_ushort, c_ubyte
import numpy as np
import cv2
import os
import tifffile
import time

class TDI:
    def __init__(self, card_ix=0, device_ix=0):
        self.card_ix = card_ix
        self.device_ix = device_ix
        self.grabber = None

    def initialize_grabber(self):
        try:
            self.release_grabber()  # Ensure no previous instance exists before initializing
            gentl = EGenTL()
            self.grabber = EGrabber(gentl, self.card_ix, self.device_ix)
            card = self.grabber.interface.get('InterfaceID')
            dev = self.grabber.device.get('DeviceID')
            width = self.grabber.remote.get('Width')
            height = self.grabber.remote.get('Height')
            
            print(f'Interface:    {card}')
            print(f'Device:       {dev}')
            print(f'Resolution:   {width}x{height}')
            print("Grabber successfully initialized.")        
            return True
        except Exception as e:
            print(f'Error initializing grabber: {e}')
            self.grabber = None
            return False

    def release_grabber(self):
        if self.grabber:
            print("Releasing grabber...")
            del self.grabber
            self.grabber = None
         
    def capture_frames_to_disc(self, N=5, outdir=None, filename_prefix="frame"):
        if not self.grabber:
            print("Grabber is not initialized. Cannot capture frames.")
            return

        if outdir is None:
            outdir = os.getcwd()  # Default to current working directory if not provided

        # Make the directory path absolute
        outdir = os.path.abspath(outdir)

        # Create the output directory if it doesn't exist
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
            print(f"Directory created: {outdir}")

        try:
            self.grabber.realloc_buffers(N)
            self.grabber.start(N)
            for frame in range(N):
                print(f"Capturing frame {frame+1}/{N}...")
                with Buffer(self.grabber, timeout=5000) as buffer:
                    rgb = buffer.convert('Mono12')
                    file_path = os.path.join(outdir, f"{filename_prefix}_{frame+1}.tiff") # Save unique filenames
                    rgb.save_to_disk(file_path)
                    print(f"Saved: {file_path}")

        except Exception as e:
            print(f"Error capturing frames: {e}")

    def display_live_frames(self, display_zoom=0.5, num_buffers=3):
        """
        Continuously captures and displays frames from the camera.
        Press any key to exit the loop.
        """
        if not self.grabber:
            print("Grabber is not initialized.")
            return

        try:
            self.grabber.realloc_buffers(num_buffers)
            self.grabber.start()

            print("Displaying live frames... Press any key in the window to stop.")

            while True:
                with Buffer(self.grabber, timeout=5000) as buffer:
                    # Get width and height
                    w = buffer.get_info(BUFFER_INFO_WIDTH, INFO_DATATYPE_SIZET)
                    h = buffer.get_info(BUFFER_INFO_DELIVERED_IMAGEHEIGHT, INFO_DATATYPE_SIZET)

                    # Convert to BGR for display
                    bgr = buffer.convert('BGR8')
                    ptr = bgr.get_address()
                    buf_size = bgr.get_buffer_size()

                    # Extract image and reshape
                    data = cast(ptr, POINTER(c_ubyte * buf_size)).contents
                    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                    img = cv2.resize(img, (int(w * display_zoom), int(h * display_zoom)))

                    # Display
                    cv2.imshow("Live Preview - Press any key to exit", img)
                    if cv2.waitKey(1) >= 0:
                        break

            cv2.destroyAllWindows()
            self.grabber.stop()
            print("Stopped live preview.")

        except Exception as e:
            print(f"Error during live frame display: {e}")

    def capture_frames_to_memory_live(self, max_frames=10, display_zoom=0.5, num_buffers=3,
                                       cal_flag=False, cal_profile=None,
                                       crop_info=None,
                                       infer_flag=False, infer_runner=None
                                       ):
        """
        Captures and displays live frames in Mono12 format, stores all intermediate images.

        Returns:
            List of dictionaries per frame with keys:
            'raw', 'calibrated', 'denoise', 'decon'
        """
        if not self.grabber:
            print("Grabber is not initialized.")
            return []

        if cal_flag and cal_profile is None:
            print("Calibration flag is set but no calibration profile provided.")
            return []

        if infer_flag and infer_runner is None:
            print("Infer flag is set but no infer_runner provided.")
            return []

        if crop_info is None:
            print("crop_info must be provided.")
            return []

        crop_width = crop_info.get('crop_width')
        center_col = crop_info.get('center_col')

        image_records = []
        try:
            self.grabber.realloc_buffers(num_buffers)
            self.grabber.start()

            print(f"Capturing up to {max_frames} frames in Mono12. Press any key in the window to stop early.")
            frame_count = 0

            while frame_count < max_frames:
                with Buffer(self.grabber, timeout=5000) as buffer:
                    w = buffer.get_info(BUFFER_INFO_WIDTH, INFO_DATATYPE_SIZET)
                    h = buffer.get_info(BUFFER_INFO_DELIVERED_IMAGEHEIGHT, INFO_DATATYPE_SIZET)

                    converted = buffer.convert('Mono12')
                    ptr = converted.get_address()
                    buf_size = converted.get_buffer_size()

                    data = cast(ptr, POINTER(c_ushort * (w * h))).contents
                    img = np.frombuffer(data, dtype=np.uint16).reshape((h, w)).astype(np.float32)
                    raw_img = img.copy()

                    if cal_flag:
                        if cal_profile.shape[0] != w:
                            print("Calibration profile width does not match image width.")
                            return []
                        calibrated_img = img / (cal_profile + 1e-8) # make sure img and cal_profile have the same astype!!!
                    else:
                        calibrated_img = img.copy()

                    # Crop image
                    start_col = max(center_col - crop_width // 2, 0)
                    end_col = start_col + crop_width
                    cropped_raw = raw_img[:, start_col:end_col]
                    cropped_cal = calibrated_img[:, start_col:end_col]

                    frame_data = {
                        'raw': np.clip(cropped_raw, 0, 65535).astype(np.uint16),
                        'calibrated': np.clip(cropped_cal, 0, 65535).astype(np.uint16)
                    }

                    if infer_flag:
                        infer_runner.raw_data = cropped_cal
                        denoise_img, decon_img = infer_runner.infer()
                        frame_data['denoise'] = denoise_img.copy()
                        frame_data['decon'] = decon_img.copy()
                        display_img = (decon_img / 256).clip(0, 255).astype(np.uint8)
                    else:
                        display_img = (cropped_cal / 16).clip(0, 255).astype(np.uint8)

                    display_img = cv2.resize(display_img, (int(display_img.shape[1] * display_zoom),
                                                           int(display_img.shape[0] * display_zoom)))
                    cv2.imshow("Mono12 Live - Press any key to stop", display_img)
                    if cv2.waitKey(1) >= 0:
                        break

                    image_records.append(frame_data)
                    frame_count += 1

            self.grabber.stop()
            print(f"Captured {len(image_records)} frames.")
            return image_records

        except Exception as e:
            print(f"Error capturing frames: {e}")
            return []

    def save_images_to_disk(self, images, outdir=None, filename_prefix="frame", keys_to_save=None):
        """
        Saves selected image types from list of image dictionaries to separate subfolders.

        Parameters:
            - images: list of dicts, each with keys like 'raw', 'calibrated', 'denoise', 'decon'
            - outdir: output directory (defaults to current working directory)
            - filename_prefix: prefix for each image file
            - keys_to_save: list of keys to save from each image dict (e.g. ['raw', 'decon'])
        """
        if not images:
            print("No images to save.")
            return

        if keys_to_save is None:
            keys_to_save = ['decon']  # Default behavior

        if outdir is None:
            outdir = os.getcwd()

        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)

        for key in keys_to_save:
            key_dir = os.path.join(outdir, key)
            os.makedirs(key_dir, exist_ok=True)

        for i, img_dict in enumerate(images):
            for key in keys_to_save:
                if key in img_dict:
                    subfolder = os.path.join(outdir, key)
                    file_path = os.path.join(subfolder, f"{filename_prefix}_{i+1}.tiff")
                    tifffile.imwrite(file_path, img_dict[key])
                    print(f"Saved: {file_path}")
                else:
                    print(f"Warning: Key '{key}' not found in frame {i+1}")


    def average_column_profile(self, folder_path, N=10):
        """
        Averages the first N TIFF files in the folder, returns a normalized 1D row (column-wise average).
        """
        tiff_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.tif', '.tiff'))
        ])[:N]

        if not tiff_files:
            print("No TIFF files found.")
            return None

        sum_image = None
        for fname in tiff_files:
            img_path = os.path.join(folder_path, fname)
            img = tifffile.imread(img_path).astype(np.float32)
            if sum_image is None:
                sum_image = img
            else:
                sum_image += img

        avg_image = sum_image / len(tiff_files)
        row_1d = np.mean(avg_image, axis=0)  # Average across rows → 1D column-wise profile
        if row_1d.max() > 0:
            row_1d /= row_1d.max()

        return row_1d