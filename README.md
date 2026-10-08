# A2D-SPI imaging pipeline
This repository contains a Python-based acquisition and reconstruction pipeline for SPI2.0. The code controls the laser, XY stage, piezoelectric Z stage, TDI camera, calibration correction, and real-time neural-network inference for denoising and deconvolution.

The pipeline is designed for acquisition-time super-resolution imaging, where TDI images are captured during continuous stage scanning, calibrated in real time, processed by a pretrained deep neural network, displayed during acquisition, and saved to disk for downstream analysis.


## Overview
The main script (3D_TDI_scan_video_test_with_buffer.ipynb) performs the following steps:

1. Initialize the OBIS laser.
2. Initialize the TDI camera and frame grabber.
3. Load a calibration profile from uniformly stained fluorescence images.
4. Initialize the MS2000 XY stage.
5. Initialize the Mad City Labs piezoelectric Z stage.
6. Load the pretrained real-time inference model.
7. Perform synchronized stage scanning and TDI image acquisition.
8. Apply real-time calibration, denoising, and deconvolution.
9. Save raw, calibrated, denoised, and deconvolved images.


## Hardware Requirements
- OBIS laser controller
- TDI camera and compatible frame grabber
- ASI MS2000 XY stage controller
- Mad City Labs Nano-Z piezoelectric Z stage
- GPU workstation for real-time inference


## Software Requirements
- numpy
- opencv-python
- imageio
- pathlib

In addition, the following custom modules must be available in the Python path:
- ms2k
- TDI
- obis_laser_controller
- madpiezo
- RT_infer

These modules provide control of the MS2000 stage, TDI camera, OBIS laser, Mad City Labs piezo stage, and real-time neural-network inference.
Locate the pretrained model in \SPI2\ZS-DeconvNet\Python_MATLAB_Codes\your_saved_models


# Contact
Hansol Yoon, Georgia Institute of Technology, h.yoon@gatech.edu
