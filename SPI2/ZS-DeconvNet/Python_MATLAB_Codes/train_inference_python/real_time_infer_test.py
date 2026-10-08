import glob
import numpy as np
import datetime
import imageio
from pathlib import Path
from typing import List
from dataclasses import dataclass
from utils.utils import prctile_norm
from models import twostage_Unet


@dataclass
class TestConfig:
    model: any
    test_num: int
    input_y: int
    input_x: int
    insert_xy: int
    upsample_flag: bool
    output_dir: Path
    weights_path: Path
    image_list: List[np.ndarray]


def load_image(image_path: str, insert_xy: int) -> np.ndarray:
    img = np.array(imageio.mimread(image_path), dtype=np.float32)
    img = np.squeeze(np.sum(img, axis=0))  # sum across channels or frames
    img = prctile_norm(img)

    # Pad symmetrically
    pad_x = np.zeros((insert_xy, img.shape[1]), dtype=np.float32)
    pad_y = np.zeros((img.shape[0] + 2 * insert_xy, insert_xy), dtype=np.float32)

    img = np.concatenate((pad_x, img, pad_x), axis=0)
    img = np.concatenate((pad_y, img, pad_y), axis=1)

    return img


def infer(config: TestConfig):
    config.model.load_weights(config.weights_path)

    for n, img in enumerate(config.image_list):
        input_shape = (1, config.input_y + 2 * config.insert_xy,
                          config.input_x + 2 * config.insert_xy, 1)
        img = img.reshape(input_shape)

        pred = config.model.predict(img)
        output1 = np.squeeze(pred[0])
        output2 = np.squeeze(pred[1])

        if config.upsample_flag:
            s = config.insert_xy
            ey, ex = config.input_y, config.input_x
            output2 = output2[2*s:2*(s+ey), 2*s:2*(s+ex)]
        else:
            s = config.insert_xy
            output2 = output2[s:s+config.input_y, s:s+config.input_x]

        output1 = np.uint16(prctile_norm(output1) * 65535)
        output2 = np.uint16(prctile_norm(output2) * 65535)

        imageio.imwrite(str(config.output_dir / f'{n}_denoised.tif'), output1)
        imageio.imwrite(str(config.output_dir / f'{n}_deconved.tif'), output2)


def main():
    # === Paths and parameters ===
    base_path = Path('C:\Users\Nikon_1\Documents\JupyterLab\hansol\ZS-DeconvNet\Python_MATLAB_Codes\your_saved_models\train_ellipses4000_twostage_Unet_Hess0.02')
    test_path = base_path / 'real_time_test'
    test_path.mkdir(parents=True, exist_ok=True)

    weight_file = base_path / 'weights_1000.h5'
    test_images = sorted(glob.glob(str(test_path / '*.tif')))
    if not test_images:
        print("No test images found.")
        return

    test_image_path = test_images[0]
    print(f"Using test image: {test_image_path}")

    # === Model and input config ===
    insert_xy, input_y, input_x = 16, 1024, 1024
    upsample_flag = True

    model = twostage_Unet.Unet((input_y + 2 * insert_xy, input_x + 2 * insert_xy, 1),
                                upsample_flag=upsample_flag,
                                insert_x=insert_xy, insert_y=insert_xy)

    padded_img = load_image(test_image_path, insert_xy)

    config = TestConfig(
        model=model,
        test_num=1,
        input_y=input_y,
        input_x=input_x,
        insert_xy=insert_xy,
        upsample_flag=upsample_flag,
        output_dir=test_path,
        weights_path=weight_file,
        image_list=[padded_img]
    )

    # === Optional: Save input reference ===
    raw_img = np.array(imageio.mimread(test_image_path), dtype=np.float32)
    raw_sum = prctile_norm(np.squeeze(np.sum(raw_img, axis=0)))
    unpadded_img = np.uint16(raw_sum * 65535)
    # imageio.imwrite(str(test_path / 'input0.tif'), unpadded_img)

    # === Warm-up + timed inference ===
    print("Warming up...")
    start = datetime.datetime.now()
    infer(config)
    print(f"Warm-up completed in: {datetime.datetime.now() - start}")

    print("Inference run...")
    start = datetime.datetime.now()
    infer(config)
    print(f"Inference completed in: {datetime.datetime.now() - start}")


if __name__ == '__main__':
    main()
