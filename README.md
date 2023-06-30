# YOLOv6_ROS

Built and tested on Python3.8, Ubuntu 20.04, ROS Noetic.

## 🧰 Build
Clone the official yolov6 repo into src directory and swap the file `yolov6/core/inferer.py` with `src/inferer.py`:

```bash
cd <yolov6_ros_directory>/src
git clone https://github.com/meituan/YOLOv6.git yolov6
git checkout 0.2.0  # Tested version works on this release only 
```

Clone the `detection_msgs` package: `https://github.com/phuoc101/detection_msgs.git`

*Note*: Please set up a ROS virtual environment with Pytorch and source it before running the package, for examnple:
```bash
virtualenv --system-site-packages -p python3.8 ~/ros_torch_env
## install dependencies in the environment base on yolov6's requirements.txt
```

To build the package:
```
cd <YOUR_ROS_WORKSPACE>
catkin build yolov6_ros
source devel/setup.bash #or setup.zsh, depends on what shell you're using
```

## 🔥 Running the demo
To run the webcam demo:

```bash
roslaunch yolov6_ros yolov6.launch
# to test with webcam, need to install usb_cam ROS package first
roslaunch yolov6_ros yolov6_webcam.launch
```
