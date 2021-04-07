#!/usr/bin/env python3
import logging
import sys
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
python3 mo.py --input_model /opt/intel/openvino_2021.3.394/ws/sample-app/models/best_model.onnx --output_dir /opt/intel/openvino_2021.3.394/ws/sample-app/models

# - m /opt/intel/openvino_2021.3.394/ws/models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml
cd /opt/intel/openvino_2021.3.394/ws/sample-app
python3 main.py \
    -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo_2.mp4  \
    -m /opt/intel/openvino_2021.3.394/ws/sample-app/models/best_model.xml  \
    -d MYRIAD 
"""
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Path to a test image file.",
                      required=True, type=str)
    args.add_argument("-m", "--model",
                      help="Required. Path to an .xml file with a pnet model.",
                      required=True, type=Path, metavar='"<path>"')
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str, metavar='"<device>"')
    args.add_argument('--loop', default=False, action='store_true',
                       help='Optional. Enable reading the input in a loop.')
    args.add_argument("--no_show",
                      help="Optional. Don't show output",
                      action='store_true')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of output to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser

def main():
    metrics = PerformanceMetrics()
    args = build_argparser().parse_args()

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")

    # IE Core
    ie = IECore()

    # Read IR
    log.info("Loading network files:\n\t{}".format(args.model))
    net = ie.read_network(args.model)

    # Input Blobs
    log.info("Preparing input blobs")
    net_input_blob = next(iter(net.input_info))

    # Output Blobs
    log.info("Preparing output blobs")
    for name, blob in net.outputs.items():
        print(f"--> Blob shape: {blob.shape}")
        print(f"--> Output layer: {name}")

    # Image Capture (RTSP-Video-Image)
    cap = open_images_capture(args.input, args.loop)
    next_frame_id = 0
    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    presenter = None
    video_writer = cv2.VideoWriter()


    # Load network
    log.info("Loading Net model to the plugin")
    w, h = (416,416)
    net.reshape({net_input_blob: [1, 3, w, h]})  # Change weidth and height of input blob
    exec_net = ie.load_network(network=net, device_name=args.device)

    while True:
        start_time = perf_counter()
        origin_image = cap.read()
        if origin_image is None:
            if next_frame_id == 0:
                raise ValueError("Can't read an image from the input")
            break
        if next_frame_id == 0:
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(origin_image.shape[1] / 4), round(origin_image.shape[0] / 8)))
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                    cap.fps(), (origin_image.shape[1], origin_image.shape[0])):
                raise RuntimeError("Can't open video writer")
        next_frame_id += 1

        rgb_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        oh, ow, _ = rgb_image.shape

        # ----------------------------------------------------------------
        # Net stage
        # ----------------------------------------------------------------
        t0 = cv2.getTickCount()
        image = cv2.resize(rgb_image, (w, h))
        image = image / 255
        image = image.transpose((2, 1, 0))
        net_input = np.expand_dims(image, axis=0)
        net_res = exec_net.infer(inputs={net_input_blob: net_input})
        #print(net_res)
        # ----------------------------------------------------------------

        infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()  # Record infer time
        cv2.putText(origin_image, 'summary: {:.1f} FPS'.format(1.0 / infer_time),(5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id <= args.output_limit - 1):
            video_writer.write(origin_image)

        if not args.no_show:
            cv2.imshow('Net Results', origin_image)
            key = cv2.waitKey(1)
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)

        metrics.update(start_time, origin_image)

    metrics.print_total()


if __name__ == '__main__':
    sys.exit(main() or 0)

