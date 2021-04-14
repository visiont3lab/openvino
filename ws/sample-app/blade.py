#!/usr/bin/env python3
import logging
import sys
import os
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from openvino.inference_engine import IECore
#sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append('/opt/intel/openvino_2021.3.394/deployment_tools/inference_engine/demos/common/python')

import monitors
from images_capture import open_images_capture
from performance_metrics import PerformanceMetrics

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

# Convert Pytorch model
# cd /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader
# python3 converter.py --name  best_model.pt  --download_dir /opt/intel/openvino_2021.3.394/ws/sample-app/models  --output_dir /opt/intel/openvino_2021.3.394/ws/sample-app/models
# 
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html

"""
# Conversion
cd /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer
python3 mo.py --input_model /opt/intel/openvino_2021.3.394/ws/sample-app/models/segnet_single.onnx --output_dir /opt/intel/openvino_2021.3.394/ws/sample-app/models
python3 mo.py --input_model /opt/intel/openvino_2021.3.394/ws/sample-app/models/segnet_dop.onnx --output_dir /opt/intel/openvino_2021.3.394/ws/sample-app/models
python3 mo.py --input_model /opt/intel/openvino_2021.3.394/ws/sample-app/models/segnet_128.onnx --output_dir /opt/intel/openvino_2021.3.394/ws/sample-app/models

# - m /opt/intel/openvino_2021.3.394/ws/models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml
cd /opt/intel/openvino_2021.3.394/ws/sample-app
python3 main.py \
    -i /opt/intel/openvino_2021.3.394/ws/sample-app/dataset/images/24745  \
    -d MYRIAD 
"""

def predict_single(init_image,net_input_blob,exec_net,model_size):
    model_h,model_w = model_size
    initial_w,initial_h = (init_image.shape[1],init_image.shape[0])
    image = cv2.resize(init_image, (model_w,model_h))
    image = image.astype(np.float32)/255.0
    x = image[np.newaxis,np.newaxis,:,:] # 1,1,512,512
    net_res = exec_net.infer(inputs={net_input_blob: x})
    mask = 255*net_res['64'][0,0,:,:]
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (initial_w,initial_h))
    rgb_image = cv2.cvtColor(init_image,cv2.COLOR_GRAY2RGB)
    rgb_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    rgb_mask[(mask>127)] = [0,255,255]
    overlay = cv2.addWeighted(rgb_image,0.8,rgb_mask,0.2,0)
    return overlay

def predict_dop(init_image,init_dop,net_input_blob,exec_net,model_size):
    model_h,model_w = model_size
    initial_w,initial_h = (init_image.shape[1],init_image.shape[0])
    image = cv2.resize(init_image, (model_w,model_h))
    image = image.astype(np.float32)/255.0
    image = image[np.newaxis,np.newaxis,:,:] # 1,1,512,512
    dop = cv2.resize(init_dop, (model_w,model_h))
    dop = dop.astype(np.float32)/255.0
    dop = dop[np.newaxis,np.newaxis,:,:] # 1,1,512,512
    x = np.concatenate((image,dop),axis=1)
    net_res = exec_net.infer(inputs={net_input_blob: x})
    mask = 255*net_res['64'][0,0,:,:]
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (initial_w,initial_h))
    rgb_image = cv2.cvtColor(init_image,cv2.COLOR_GRAY2RGB)
    rgb_dop = cv2.cvtColor(init_dop,cv2.COLOR_GRAY2RGB)
    rgb_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    rgb_mask[(mask>127)] = [0,255,255]
    overlay = cv2.addWeighted(rgb_image,0.8,rgb_mask,0.2,0)
    return overlay

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Path to a test image file.",
                      required=True, type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str, metavar='"<device>"')
    return parser

def main():
    metrics = PerformanceMetrics()
    args = build_argparser().parse_args()

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")

    # IE Core
    ie = IECore()

    # Parameters
    path2testImages = os.path.join(args.input,"images")
    path2testDop = os.path.join(args.input,"dop")

    #path2model = "/opt/intel/openvino_2021.3.394/ws/sample-app/models/segnet_dop.xml"
    path2model = "/opt/intel/openvino_2021.3.394/ws/sample-app/models/segnet_128.xml"
    model_c , model_w, model_h = (2, 128,128)

    #path2model = "/opt/intel/openvino_2021.3.394/ws/sample-app/models/segnet_single.xml"
    #model_c , model_w, model_h = (1, 512,512)

    # Load network Read IR
    log.info("Loading network files:\n\t{}".format(path2model))
    net = ie.read_network(path2model)

    # Input Blobs
    log.info("Preparing input blobs")
    net_input_blob = next(iter(net.input_info))

    # Output Blobs
    log.info("Preparing output blobs")
    for name, blob in net.outputs.items():
        print(f"--> Blob shape: {blob.shape}")
        print(f"--> Output layer: {name}")

    log.info("Loading Net model to the plugin")
    net.reshape({net_input_blob: [1, model_c, model_w, model_h]})  # Change weidth and height of input blob
    exec_net = ie.load_network(network=net, device_name=args.device)

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    names = os.listdir(path2testImages)
    sum_time_sec = 0 
    for name in names:
        extension = os.path.splitext(name)[1]
        if extension==".png":
   
            start_time = perf_counter()    
            t0 = cv2.getTickCount()
           
            # Reading
            filename = os.path.join(path2testImages,name)
            init_image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            filename_dop = os.path.join(path2testDop,name.replace("_Avg","_Dop"))
            init_dop = cv2.imread(filename_dop,cv2.IMREAD_GRAYSCALE)
            
            # ----------------------------------------------------------------
            # Net stage
            # ----------------------------------------------------------------
            if model_c==1:
                overlay = predict_single(init_image,net_input_blob,exec_net,(model_h,model_w))
            else:
                overlay = predict_dop(init_image,init_dop,net_input_blob,exec_net,(model_h,model_w))
            # ----------------------------------------------------------------

            infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()  # Record infer time
            cv2.putText(overlay, 'summary: {:.1f} FPS'.format(1.0 / infer_time),(5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))
            cv2.putText(overlay, 'inference time: {:.4f}s'.format(infer_time),(5, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))

            cv2.imshow('Net Results', overlay)
            key = cv2.waitKey(1)
            if key in {ord('q'), ord('Q'), 27}:
                break

        metrics.update(start_time, overlay)

    metrics.print_total()


if __name__ == '__main__':
    sys.exit(main() or 0)

