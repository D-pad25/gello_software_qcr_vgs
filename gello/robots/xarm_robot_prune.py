import dataclasses
import threading
import time
from typing import Dict, Optional

import numpy as np
from pyquaternion import Quaternion

from gello.robots.prune_robot import Robot
from experiments import prune_sub
# Add subcriber module for prune comms
# from vgs_robot.vgs_perception_n_grasp.src.subscribers.xarm_sub import XarmSubscriber
import rospy
import std_msgs.msg


def _aa_from_quat(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to an axis-angle representation.

    Args:
        quat (np.ndarray): The quaternion to convert.

    Returns:
        np.ndarray: The axis-angle representation of the quaternion.
    """
    assert quat.shape == (4,), "Input quaternion must be a 4D vector."
    norm = np.linalg.norm(quat)
    assert norm != 0, "Input quaternion must not be a zero vector."
    quat = quat / norm  # Normalize the quaternion

    Q = Quaternion(w=quat[3], x=quat[0], y=quat[1], z=quat[2])
    angle = Q.angle
    axis = Q.axis
    aa = axis * angle
    return aa


def _quat_from_aa(aa: np.ndarray) -> np.ndarray:
    """Convert an axis-angle representation to a quaternion.

    Args:
        aa (np.ndarray): The axis-angle representation to convert.

    Returns:
        np.ndarray: The quaternion representation of the axis-angle.
    """
    assert aa.shape == (3,), "Input axis-angle must be a 3D vector."
    norm = np.linalg.norm(aa)
    assert norm != 0, "Input axis-angle must not be a zero vector."
    axis = aa / norm  # Normalize the axis-angle

    Q = Quaternion(axis=axis, angle=norm)
    quat = np.array([Q.x, Q.y, Q.z, Q.w])
    return quat


@dataclasses.dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    z: float
    prune: float
    j1: float
    j2: float
    j3: float
    j4: float
    j5: float
    j6: float
    aa: np.ndarray

    @staticmethod
    def from_robot(
        cartesian: np.ndarray,
        joints: np.ndarray,
        prune: float,
        aa: np.ndarray,
    ) -> "RobotState":
        return RobotState(
            cartesian[0],
            cartesian[1],
            cartesian[2],
            prune,
            joints[0],
            joints[1],
            joints[2],
            joints[3],
            joints[4],
            joints[5],
            aa,
        )

    def cartesian_pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def quat(self) -> np.ndarray:
        return _quat_from_aa(self.aa)

    def joints(self) -> np.ndarray:
        return np.array([self.j1, self.j2, self.j3, self.j4, self.j5, self.j6])

    def prune_pos(self) -> float:
        return self.prune


class Rate:
    def __init__(self, *, duration):
        self.duration = duration
        self.last = time.time()

    def sleep(self, duration=None) -> None:
        duration = self.duration if duration is None else duration
        assert duration >= 0
        now = time.time()
        passed = now - self.last
        remaining = duration - passed
        assert passed >= 0
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()

######################################################################
####################### CUSTOM PRUNE XARM CLASS ####################
######################################################################
class XArmRobotPrune(Robot):
    #sets max speed to joint servo error
    DEFAULT_MAX_DELTA = 0.01
    
    def num_dofs(self) -> int:
        return 7

    def get_joint_state(self) -> np.ndarray:
        state = self.get_state()
        pruner = state.prune_pos()
        all_dofs = np.concatenate([state.joints(), np.array([pruner])])
        return all_dofs

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        if len(joint_state) == 6:
            self.set_command(joint_state, None)
        elif len(joint_state) == 7:
            self.set_command(joint_state[:6], joint_state[6])
        else:
            raise ValueError(
                f"Invalid joint state: {joint_state}, len={len(joint_state)}"
            )

    def stop(self):
        self.running = False
        if self.robot is not None:
            self.robot.disconnect()

        if self.command_thread is not None:
            self.command_thread.join()

    def __init__(
        self,
        ip: str = "192.168.1.226",
        real: bool = True,
        control_frequency: float = 30.0,
        max_delta: float = DEFAULT_MAX_DELTA,
    ):
        print(ip)
        self.real = real
        self.max_delta = max_delta
        if real:
            from xarm.wrapper import XArmAPI

            self.robot = XArmAPI(ip, is_radian=True)
        else:
            self.robot = None

        # self.xarm_sub = XarmSubscriber()
        print("Calling rospy.init_node")
        rospy.init_node('xarm_gello_node')
        print("After rospy.init_node")
        self.prune_pose_sub = prune_sub.PruneSubscriber()
        self.prune_pose_pub = rospy.Publisher('/prune_command', std_msgs.msg.Int16, queue_size=10)

        self._control_frequency = control_frequency
        self._clear_error_states()

        self.last_state_lock = threading.Lock()
        self.target_command_lock = threading.Lock()
        self.last_state = self._update_last_state()
        self.target_command = {
            "joints": self.last_state.joints(),
            "prune": 0,
        }
        self.running = True
        self.command_thread = None
        if real:
            self.command_thread = threading.Thread(target=self._robot_thread)
            self.command_thread.start()

    def get_state(self) -> RobotState:
        with self.last_state_lock:
            return self.last_state

    def set_command(self, joints: np.ndarray, prune: Optional[float] = None) -> None:
        with self.target_command_lock:
            self.target_command = {
                "joints": joints,
                "prune": prune,
            }

    def _clear_error_states(self):
        if self.robot is None:
            return
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.motion_enable(True)
        time.sleep(1)
        self.robot.set_mode(1)
        time.sleep(1)
        self.robot.set_collision_sensitivity(0)
        time.sleep(1)
        self.robot.set_state(state=0)
        time.sleep(1)
        # self.robot.set_prune_enable(True)
        time.sleep(1)
        # self.robot.set_prune_mode(0)
        time.sleep(1)
        # self.robot.set_prune_speed(3000)
        time.sleep(1)

    def _set_prune_position(self, pos: int) -> None:
        # print(f'POS = {pos}')
        if self.robot is None:
            return        
        raw_scaled_value = pos * 255

        # Clamp value
        clamped_value = max(0, min(255, int(round(raw_scaled_value))))  
        # print(f"DEBUG ---> CLAMPED VALUE: {clamped_value}")
        # Prepare to send message
        msg = std_msgs.msg.Int16()
        if clamped_value < 100:
            msg.data = 0
        elif (clamped_value > 100) and (clamped_value < 200):
            msg.data = 1
        else:
            msg.data = 2
        # print(f'DEBUG ---> OUTPUT: {msg.data}')

        # Send message
        self.prune_pose_pub.publish(msg)

    def _robot_thread(self):
        rate = Rate(
            duration=1 / self._control_frequency
        )  # command and update rate for robot
        step_times = []
        count = 0
        while self.running:
            s_t = time.time()
            # update last state
            self.last_state = self._update_last_state()
            with self.target_command_lock:
                joint_delta = np.array(
                    self.target_command["joints"] - self.last_state.joints()
                )
                prune_command = self.target_command["prune"]

            norm = np.linalg.norm(joint_delta)

            # threshold delta to be at most 0.01 in norm space
            if norm > self.max_delta:
                delta = joint_delta / norm * self.max_delta
            else:
                delta = joint_delta

            # command position
            self._set_position(
                self.last_state.joints() + delta,
            )

            if prune_command is not None:
                set_point = prune_command
                self._set_prune_position(set_point)
            self.last_state = self._update_last_state()

            rate.sleep()
            step_times.append(time.time() - s_t)
            count += 1
            if count % 1000 == 0:
                # Mean, Std, Min, Max, only show 3 decimal places and string pad with 10 spaces
                frequency = 1 / np.mean(step_times)
                # print(f"Step time - mean: {np.mean(step_times):10.3f}, std: {np.std(step_times):10.3f}, min: {np.min(step_times):10.3f}, max: {np.max(step_times):10.3f}")
                print(
                    f"Low  Level Frequency - mean: {frequency:10.3f}, std: {np.std(frequency):10.3f}, min: {np.min(frequency):10.3f}, max: {np.max(frequency):10.3f}"
                )
                step_times = []

    def _update_last_state(self) -> RobotState:
        with self.last_state_lock:
            if self.robot is None:
                return RobotState(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.zeros(3))           
            
            prune_pos = self.prune_pose_sub.get_latest_position()

            code, servo_angle = self.robot.get_servo_angle(is_radian=True)
            while code != 0:
                print(f"Error code {code} in get_servo_angle().")
                self._clear_error_states()
                code, servo_angle = self.robot.get_servo_angle(is_radian=True)

            code, cart_pos = self.robot.get_position_aa(is_radian=True)
            while code != 0:
                print(f"Error code {code} in get_position().")
                self._clear_error_states()
                code, cart_pos = self.robot.get_position_aa(is_radian=True)

            cart_pos = np.array(cart_pos)
            aa = cart_pos[3:]
            cart_pos[:3] /= 1000

            return RobotState.from_robot(
                cart_pos,
                servo_angle,
                prune_pos,
                aa,
            )

    def _set_position(
        self,
        joints: np.ndarray,
    ) -> None:
        if self.robot is None:
            return
        # threhold xyz to be in  min max
        ret = self.robot.set_servo_angle_j(joints, wait=False, is_radian=True)
        if ret in [1, 8]:
            self._clear_error_states()

    def get_observations(self) -> Dict[str, np.ndarray]:
        state = self.get_state()
        pos_quat = np.concatenate([state.cartesian_pos(), state.quat()])
        joints = self.get_joint_state()
        return {
            "joint_positions": joints,  # rotational joint + prune state
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "prune_position": np.array(state.prune_pos()),
        }

def main():
    ip = "192.168.1.226"
    robot = XArmRobotPrune(ip)
    import time

    time.sleep(1)
    print(robot.get_state())

    time.sleep(1)
    print(robot.get_state())
    print("end")
    robot.stop()


if __name__ == "__main__":
    main()
