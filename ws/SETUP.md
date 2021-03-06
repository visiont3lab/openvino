# OpenVino Docker setup for inference on NCS2

```
# Explaination Website: https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_docker_linux.html#use_docker_image_for_cpu
# Dockerfile: https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles

# We are intersted in Making Inference on Neural Compute stick 2 (NCS2) by using Plugin Myriad on Ubuntu 18.04
# Dockerhub: https://hub.docker.com/r/openvino/ubuntu18_data_dev
# docker run  --help  --> to see options
xhost +local:docker && \
docker run --rm --name openvino_ncs2_dev  -it \
        -u 0 \
        --device /dev/dri:/dev/dri \
        --device-cgroup-rule='c 189:* rmw' -v \
        ~/.Xauthority:/root/.Xauthority   \
        -v /dev/bus/usb:/dev/bus/usb \
        -v /home/visionlab/Documents/openvino/ws:/opt/intel/openvino_2021.3.394/ws \
        --network host \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
        -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
        -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native:Z \
         visiont3lab/openvino-ubuntu18.04 \
        /bin/bash

# visiont3lab/openvino-ubuntu18.04 --> compiled version
# openvino/ubuntu18_data_dev:latest --> initial versione
```

* [Get Started Guide](https://docs.openvinotoolkit.org/latest/openvino_docs_get_started_get_started_linux.html)

### Demo apps

```
# Demo apps!! Do this inside docker
apt update && apt install sudo && apt install libnotify-dev vim
cd /deployment_tools/demo
# ./<script_name> -d [CPU, GPU, MYRIAD, HDDL]
./demo_squeezenet_download_convert_run.sh -d MYRIAD   # CPU
./demo_security_barrier_camera.sh -d MYRIAD           #-sample-options -no_show
./demo_speech_recognition.sh -d MYRIAD
./demo_benchmark_app.sh -d MYRIAD
```

### [Demo Applications](https://docs.openvinotoolkit.org/latest/omz_demos.html)

```
# get samples iot video (outside docker container)
cd open_vino/ws
git clone https://github.com/intel-iot-devkit/sample-videos
# inside docker container

# --- Build demos c++/ python
# https://docs.openvinotoolkit.org/latest/omz_demos.html
# https://docs.openvinotoolkit.org/2021.2/omz_demos_README.html
cd /opt/intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos
# --> Open ./build_demos.sh  and add change line 84 --> (cd "$build_dir" && cmake  -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release "${extra_cmake_opts[@]}" "$DEMOS_PATH")
# add -DENABLE_PYTHON=ON
# Build demo python
https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/New-3D-human-pose-estimation-demo/td-p/1183637
./build_demo.sh
## add to bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/root/omz_demos_build/intel64/Release/lib' >> $HOME/.bashrc

# --- Run human_pose_estimation_3d_demo
# human_pose_estimation_3d_demo https://docs.openvinotoolkit.org/latest/omz_models_model_human_pose_estimation_3d_0001.html
# human_pose_estimation_3d_demo https://docs.openvinotoolkit.org/latest/omz_demos_human_pose_estimation_3d_demo_python.html
# Dowload and convert model  https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html
# Video Explaination: https://www.youtube.com/watch?v=4LAAjEzh2nU
cd /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader
python3 downloader.py --print_all
python3 downloader.py --name human-pose-estimation-3d-0001 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 converter.py --name  human-pose-estimation-3d-0001  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models

cd /opt/intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos/human_pose_estimation_3d_demo/python
python3 human_pose_estimation_3d_demo.py \
    -m /opt/intel/openvino_2021.3.394/ws/models/public/human-pose-estimation-3d-0001/FP16/human-pose-estimation-3d-0001.xml \
    -i /opt/intel/openvino_2021.3.394/ws/sample-videos/face-demographics-walking.mp4 \
    -d MYRIAD


# --- Run gaze_estimation_demo
# https://docs.openvinotoolkit.org/latest/omz_demos_gaze_estimation_demo_cpp.html
cd /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader
python3 downloader.py --name gaze-estimation-adas-0002 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name face-detection-retail-0004 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name face-detection-adas-0001 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name head-pose-estimation-adas-0001 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name facial-landmarks-35-adas-0002 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name open-closed-eye-0001 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 converter.py --name open-closed-eye-0001  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models

cd /root/omz_demos_build/intel64/Release
# -i /opt/intel/openvino_2021.3.394/ws/sample-videos/head-pose-face-detection-female.mp4
# -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo.mp4
# -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo_1.mp4
# -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo_2.mp4
# -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo_3.mp4

./gaze_estimation_demo -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo.mp4 \
    -m    /opt/intel/openvino_2021.3.394/ws/models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \
    -m_fd /opt/intel/openvino_2021.3.394/ws/models/intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml \
    -m_hp /opt/intel/openvino_2021.3.394/ws/models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \
    -m_lm /opt/intel/openvino_2021.3.394/ws/models/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml \
    -m_es /opt/intel/openvino_2021.3.394/ws/models/public/open-closed-eye-0001/FP16/open-closed-eye-0001.xml \
    -d MYRIAD \
    -d_fd MYRIAD \
    -d_lm MYRIAD \
    -d_es MYRIAD

# Try to press g,b,o,l,e,a,c,n,f,esc

## --- Security Barrier Camera C++ Demo
# https://docs.openvinotoolkit.org/latest/omz_demos_security_barrier_camera_demo_cpp.html
cd /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader
# New models (public not good results)
python3 downloader.py --name vehicle-license-plate-detection-barrier-0123 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name vehicle-attributes-recognition-barrier-0042 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name license-plate-recognition-barrier-0007 --output_dir /opt/intel/openvino_2021.3.394/ws/models
# Offical Intel models tested on vpu (ncs2)
python3 downloader.py --name vehicle-license-plate-detection-barrier-0106 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name vehicle-attributes-recognition-barrier-0039 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name license-plate-recognition-barrier-0001 --output_dir /opt/intel/openvino_2021.3.394/ws/models
# conversion
python3 converter.py --name  license-plate-recognition-barrier-0007  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 converter.py --name vehicle-license-plate-detection-barrier-0123  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models

cd /root/omz_demos_build/intel64/Release
./security_barrier_camera_demo \
    -i     /opt/intel/openvino_2021.3.394/ws/sample-videos/parking.mp4  /opt/intel/openvino_2021.3.394/ws/sample-videos/car-detection.mp4 \
    -m     /opt/intel/openvino_2021.3.394/ws/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml \
    -m_va  /opt/intel/openvino_2021.3.394/ws/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml \
    -m_lpr /opt/intel/openvino_2021.3.394/ws/models/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.xml \
    -d MYRIAD \
    -d_va MYRIAD \
    -d_lpr MYRIAD
# -display_resolution 1280x720

## --- Multi channel Human pose estimation (up to 16 camera s inpus) NOTE: UNABLE TO MAKE IT RUN WITH MULTIPLES VIDEO (nc flag not found!)
# https://docs.openvinotoolkit.org/latest/omz_demos_multi_channel_human_pose_estimation_demo_cpp.html
# https://www.youtube.com/watch?v=2G6uSHPFP-Q
python3 downloader.py --name  human-pose-estimation-0001 --output_dir /opt/intel/openvino_2021.3.394/ws/models

cd /root/omz_demos_build/intel64/Release
./multi_channel_human_pose_estimation_demo \
    -m /opt/intel/openvino_2021.3.394/ws/models/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml \
    -i /opt/intel/openvino_2021.3.394/ws/sample-videos/head-pose-face-detection-female.mp4 /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo.mp4 \
    -nc 2 \
    -nireq 2  \
    -bs 2 \
    -show_stats \
    -d MYRIAD

## -- Pedestrian Tracker
python3 downloader.py --print_all | grep rei
cd /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader
python3 downloader.py --name  person-detection-retail-0013 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name  person-reidentification-retail-0277 --output_dir /opt/intel/openvino_2021.3.394/ws/models

cd /root/omz_demos_build/intel64/Release
# -i /opt/intel/openvino_2021.3.394/ws/sample-videos/face-demographics-walking-and-pause.mp4
./pedestrian_tracker_demo \
    -m_det /opt/intel/openvino_2021.3.394/ws/models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml \
    -m_reid /opt/intel/openvino_2021.3.394/ws/models/intel/person-reidentification-retail-0277/FP16/person-reidentification-retail-0277.xml \
    -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_pedestrian_tracker.mp4  \
    -d_det CPU \
    -d_reid CPU

## -- Object detection
# https://docs.openvinotoolkit.org/latest/omz_models_group_public.html

python3 downloader.py --print_all | grep rei
cd /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader
# Download all object detection models
python3 downloader.py --list /opt/intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos/people-remove/python/models.lst
python3 downloader.py --name  pedestrian-and-vehicle-detector-adas-0001 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name  yolo-v3-tiny-tf --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name  yolo-v3-tf --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name faceboxes-pytorch --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name retinaface-resnet50 --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 downloader.py --name faster_rcnn_inception_resnet_v2_atrous_coco --output_dir /opt/intel/openvino_2021.3.394/ws/models

# Convert
# https://www.youtube.com/watch?v=wmRNqg_7Eo0&list=PLg-UKERBljNzXUIDjeb8oF-KRwp2fTU4i&index=70
python3 converter.py --name yolo-v3-tiny-tf  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 converter.py --name yolo-v3-tf  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 converter.py --name ssd512  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 converter.py --name faceboxes-pytorch  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models
python3 converter.py --name faster_rcnn_inception_resnet_v2_atrous_coco  --download_dir /opt/intel/openvino_2021.3.394/ws/models  --output_dir /opt/intel/openvino_2021.3.394/ws/models

cd /opt/intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos/object_detection_demo/python
-m /opt/intel/openvino_2021.3.394/ws/models/public/yolo-v3-tiny-tf/FP32/yolo-v3-tiny-tf.xml  -at yolo
-m /opt/intel/openvino_2021.3.394/ws/models/public/yolo-v3-tf/FP32/yolo-v3-tf.xml  -at yolo   --input_size 608 608
-m /opt/intel/openvino_2021.3.394/ws/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -at ssd --input_size 320 544

# Object detection fast
python3 object_detection_demo.py \
    -i /opt/intel/openvino_2021.3.394/ws/sample-videos//rome-1920-1080-10fps-short.mp4 \
    -m /opt/intel/openvino_2021.3.394/ws/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -at ssd --input_size 320 544 \
    -d MYRIAD 

# Object detection ok 
python3 object_detection_demo.py \
    -i /opt/intel/openvino_2021.3.394/ws/sample-videos//rome-1920-1080-10fps-short.mp4 \
    -m /opt/intel/openvino_2021.3.394/ws/models/public/yolo-v3-tf/FP32/yolo-v3-tf.xml  -at yolo   --input_size 608 608  \
    -d MYRIAD 

# Face detection 2fps faceboxes
python3 object_detection_demo.py \
    -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo_3.mp4 \
    -m /opt/intel/openvino_2021.3.394/ws/models/public/faceboxes-pytorch/FP32/faceboxes-pytorch.xml -at faceboxes --input_size 1024 1024 \
    -d MYRIAD 

# Face detection 1fps retinaface-50
python3 object_detection_demo.py \
    -i /opt/intel/openvino_2021.3.394/ws/sample-videos/manuel_gaze_estimation_demo_3.mp4 \
    -m /opt/intel/openvino_2021.3.394/ws/models/public/retinaface-resnet50/FP32/retinaface-resnet50.xml -at retinaface --input_size 640 640 \
    -d MYRIAD 

# -- DL streamer
cd /opt/intel/openvino_2021.3.394/data_processing/dl_streamer/samples/gst_launch/face_detection_and_classification

```

## Commit created image

```
2674cfec1180 is the container ID
docker commit -m "openvino-samples-test" 2674cfec1180 visiont3lab/openvino-ubuntu18.04
docker push visiont3lab/openvino-ubuntu18.04:latest
```
## Workbench Docker https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Install_from_Docker_Hub.html


[Intro intel openvino example](https://towardsdatascience.com/a-quick-intro-to-intels-openvino-toolkit-for-faster-deep-learning-inference-d695c022c1ce)
