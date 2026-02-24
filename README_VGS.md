# VGS Lab Data Collection (Gello) — Tomato Gripper & Pruning Tool
This repository branch provides procedures and tooling for collecting imitation-learning datasets using the GELLO framework with the xArm robot, tomato gripper, and pruning tool in the VGS lab environment.

## Table of contents
- [System overview](#system-overview)
- [Quick start](#quick-start)
- [Setup](#setup)
  - [1.0 ROS + rosserial prerequisites](#10-ros--rosserial-prerequisites)
  - [1.1 End-effector enable/disable (important)](#11-end-effector-enabledisable-important)
  - [2.0 Camera setup (ZMQ)](#20-camera-setup-zmq)
    - [2.1 Configure the number of cameras](#21-configure-the-number-of-cameras)
    - [2.2 Gello environment camera dictionary](#22-gello-environment-camera-dictionary)
    - [2.3 Camera client configuration](#23-camera-client-configuration)
    - [2.4 Launch camera nodes](#24-launch-camera-nodes)
    - [2.5 Launch camera clients (optional)](#25-optional-launch-camera-clients-live-preview)
  - [3.0 Robot node setup](#30-robot-node-setup)
    - [3.1 Launch robot nodes](#31-launch-robot-nodes)
  - [4.0 Run the Gello environment](#40-run-the-gello-environment)
    - [4.1 Tomato gripper](#41-tomato-gripper)
    - [4.2 Pruning tool](#42-pruning-tool)
    - [4.3 Data directory (when saving)](#43-data-directory-when-saving)
- [Troubleshooting](#troubleshooting)

---

## System overview

A typical data-collection run looks like:

1. Start ROS and the rosserial node (for end-effector commands).
2. Enable the attached end-effector(s) (tomato gripper / pruning tool).
3. Launch camera nodes (ZMQ servers), and optionally camera clients for live preview.
4. Launch robot nodes for the chosen tool (tomato vs pruning).
5. Run Gello with optional saving enabled.


---

## Quick start
>**NOTE:** Quick start assumes you have read and familiarised yourself with [Setup](#setup)

From the repository root:

1) **Start ROS**
```bash
roscore
```

2) **Start rosserial**
```bash
rosrun rosserial_python serial_node.py __name:=stepteensy_serial_node _port:=/dev/ttyACM0 _baud:=115200
```

3) **Enable the correct end-effector** (see details below)

4) **Launch camera nodes**
```bash
python experiments/launch_camera_nodes.py
```

5) *(Optional)* **Launch camera clients** (live preview)
```bash
python experiments/launch_camera_clients.py
```

6) **Launch robot nodes**
- Tomato gripper:
  ```bash
  python experiments/launch_nodes.py --robot xarm
  ```
- Pruning tool:
  ```bash
  python experiments/launch_nodes.py --robot xarm_prune
  ```

7) **Run Gello**
- Tomato (no saving):
  ```bash
  python experiments/run_env.py --agent=gello
  ```
- Tomato (save data):
  ```bash
  python experiments/run_env.py --agent=gello --use-save-interface
  ```
- Pruning (no saving):
  ```bash
  python experiments/run_env.py --agent=gello --prune
  ```
- Pruning (save data):
  ```bash
  python experiments/run_env.py --agent=gello --prune --use-save-interface
  ```

---

## Setup

### 1.0 ROS + rosserial prerequisites

This environment uses a ROS rosserial node to send commands to the end effectors (tomato gripper / pruning tool). Before launching any robot scripts, ensure:

1. `roscore` is running
2. The `rosserial_python` serial node is running (example below)

```bash
rosrun rosserial_python serial_node.py __name:=stepteensy_serial_node _port:=/dev/ttyACM0 _baud:=115200
```

---

#### **1.1 End-effector enable/disable (important)**

There is a known issue: if either the pruning tool or tomato gripper is NOT attached to the xArm, the motor controller may freeze. 
To avoid this, disable any end-effector that is not physically attached.

Disable:
```bash
rostopic pub -1 /gripper_enable std_msgs/Int16 "data: 0"
rostopic pub -1 /prune_enable   std_msgs/Int16 "data: 0"
```

Enable:
```bash
rostopic pub -1 /gripper_enable std_msgs/Int16 "data: 1"
rostopic pub -1 /prune_enable   std_msgs/Int16 "data: 1"
```

> **Note:** This step is vital—if end-effectors are not correctly enabled/disabled, you may be unable to send/receive commands reliably.

---

### 2.0 Camera setup (ZMQ)

The camera pipeline uses ZMQ server/client communication. You will typically:
1. Configure which cameras are used
2. Launch camera nodes (servers)
3. *(Optional)* Launch camera clients for live preview

#### **2.1 Configure the number of cameras**

Two locations must match your intended camera configuration:

1) **Gello environment configuration**
- `gello/env.py` (tomato)
- `gello/prune_env.py` (pruning)

2) **Camera clients**
- `experiments/launch_camera_clients.py`

#### **2.2 Gello environment camera dictionary**

The environment determines the number of cameras via a camera dictionary (`camera_dict`) populated in `experiments/run_env.py` (inside `main`):

```python
camera_clients = {
    "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
    "base":  ZMQClientCamera(port=args.base_camera_port,  host=args.hostname),
}
```

To change the number of cameras:
- Add/remove entries in this dictionary (e.g., comment out `"base"` if only using a wrist camera).

#### **2.3 Camera client configuration**

`experiments/launch_camera_clients.py` includes instructions for configuring multiple cameras and ports. Ensure the clients match the cameras you enabled in `camera_clients`.


#### **2.4 Launch camera nodes**

From the repository root:
```bash
python experiments/launch_camera_nodes.py
```

#### **2.5 (Optional) Launch camera clients (live preview)**

Live preview is strongly recommended during data collection:

```bash
python experiments/launch_camera_clients.py
```

> Ensure camera nodes are running before starting clients.

---

### 3.0 Robot node setup

Robot nodes must be launched after:
- ROS + rosserial prerequisites are running, and
- End-effector enable/disable is set correctly.

#### **3.1 Launch robot nodes**
Pending on what end effector you are using, run one of the following commands:
- Tomato gripper:
  ```bash
  python experiments/launch_nodes.py --robot xarm
  ```
- Pruning tool:
  ```bash
  python experiments/launch_nodes.py --robot xarm_prune
  ```

---

### 4.0 Run the Gello environment

Once cameras and robot nodes are running, start the environment:

#### **4.1 Tomato gripper**
- No saving:
  ```bash
  python experiments/run_env.py --agent=gello
  ```
- Save data:
  ```bash
  python experiments/run_env.py --agent=gello --use-save-interface
  ```

#### **4.2 Pruning tool**
- No saving:
  ```bash
  python experiments/run_env.py --agent=gello --prune
  ```
- Save data:
  ```bash
  python experiments/run_env.py --agent=gello --prune --use-save-interface
  ```

#### **4.3 Data directory (when saving)**

If you use the save interface, ensure the dataset output path is set correctly.

- Update the `data_dir` path in `experiments/run_env.py` to your desired location.

---

## Troubleshooting

### End-effector controller “freezes”
- Confirm **only attached** tools are enabled.
- Disable the unused tool:
  ```bash
  rostopic pub -1 /gripper_enable std_msgs/Int16 "data: 0"
  rostopic pub -1 /prune_enable   std_msgs/Int16 "data: 0"
  ```

### rosserial can’t open the port
- Verify the port exists:
  ```bash
  ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
  ```
- Check permissions (common on Ubuntu):
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  Log out/in after changing groups.

### Camera clients show a black screen / no frames
- Ensure camera nodes are running **first**.
- Verify ports/hostnames match between:
  - `experiments/launch_camera_nodes.py`
  - `experiments/run_env.py` (camera dictionary)
  - `experiments/launch_camera_clients.py`

### ROS topics not publishing
- Inspect topics:
  ```bash
  rostopic list
  rostopic echo /gripper_enable
  rostopic echo /prune_enable
  ```