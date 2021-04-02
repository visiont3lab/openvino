
# OpenVino Docker setup for inference on NCS2

> The image is [visiont3lab/openvino-ubuntu18.04](https://hub.docker.com/repository/docker/visiont3lab/openvino-ubuntu18.04)

## Get video

```
# -- Get samples iot video (outside docker container)
cd openvino/ws
git clone https://github.com/intel-iot-devkit/sample-videos
```

## Run container

```
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
```

##  Demo apps

```
# -- Run inside docker container
cd /deployment_tools/demo
# ./<script_name> -d [CPU, GPU, MYRIAD, HDDL]
./demo_squeezenet_download_convert_run.sh -d MYRIAD   # CPU
./demo_security_barrier_camera.sh -d MYRIAD           #-sample-options -no_show
./demo_speech_recognition.sh -d MYRIAD
./demo_benchmark_app.sh -d MYRIAD
```

##  [Demo Applications](https://docs.openvinotoolkit.org/latest/omz_demos.html)


### Run human_pose_estimation_3d_demo

```
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
```

### Run gaze_estimation_demo

```
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
```

### Security Barrier Camera C++ Demo

```
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
```

##  Multi channel Human pose estimation (up to 16 camera s inpus) 

> NOTE: UNABLE TO MAKE IT RUN WITH MULTIPLES VIDEO (nc flag not found!)

```
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
```

## Pedestrian Tracker

```
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

```