# RT_infer.py

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices available:", tf.config.list_physical_devices('GPU'))
print("TF built with CUDA:", tf.sysconfig.get_build_info()["cuda_version"])

import numpy as np
from pathlib import Path
from typing import List, Union
import sys
utils_path = Path("ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python")
sys.path.append(str(utils_path))
from utils.utils import prctile_norm
from utils.utils import fixed_norm
from models import twostage_Unet


class RTInfer:
    def __init__(self,
                 model_weights_path: Path,
                 raw_data: Union[np.ndarray, List[np.ndarray]],
                 input_shape: tuple = (1024, 1024),
                 insert_xy: int = 16,
                 upsample_flag: bool = True):
        
        print("Initializing model...")

        self.weights_path = model_weights_path
        self.raw_data = raw_data if isinstance(raw_data, list) else [raw_data]
        self.input_y, self.input_x = input_shape
        self.insert_xy = insert_xy
        self.upsample_flag = upsample_flag

        self.model = twostage_Unet.Unet(
            (self.input_y + 2 * self.insert_xy, self.input_x + 2 * self.insert_xy, 1),
            upsample_flag=self.upsample_flag,
            insert_x=self.insert_xy,
            insert_y=self.insert_xy
        )
        self.model.load_weights(self.weights_path)
        
        # test dummy inference
        print("Running dummy inference to warm up model...")
        _ = self.infer()
        print("Initialization complete.")

    def _pad_and_normalize(self, img: np.ndarray) -> np.ndarray:
        img = np.squeeze(self.raw_data)
        print(f"[DEBUG] Raw image stats before normalization: min={img.min()}, max={img.max()}, mean={img.mean()}")

        #img = prctile_norm(img)
        img = fixed_norm(img, min_val=0, max_val=9077)
        print("Input image shape before padding:", img.shape)

        pad_x = np.zeros((self.insert_xy, img.shape[1]), dtype=np.float32)
        pad_y = np.zeros((img.shape[0] + 2 * self.insert_xy, self.insert_xy), dtype=np.float32)

        img = np.concatenate((pad_x, img, pad_x), axis=0)
        img = np.concatenate((pad_y, img, pad_y), axis=1)

        return img

    def infer(self, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            output1 (denoised image), output2 (deconvolved image)
        """
        img = self._pad_and_normalize(self.raw_data)
        input_tensor = img.reshape(1, img.shape[0], img.shape[1], 1)

        pred = self.model.predict(input_tensor, verbose=1)
        output1 = np.squeeze(pred[0])
        output2 = np.squeeze(pred[1])

        s = self.insert_xy
        if self.upsample_flag:
            ey, ex = self.input_y, self.input_x
            output2 = output2[2 * s:2 * (s + ey), 2 * s:2 * (s + ex)]
        else:
            output2 = output2[s:s + self.input_y, s:s + self.input_x]

        output1 = np.uint16(np.clip(output1, 0, 1) * 65535)
        output2 = np.uint16(np.clip(output2, 0, 1) * 65535)

        return output1, output2
