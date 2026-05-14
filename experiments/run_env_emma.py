import glob
import time
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from collections import deque
import numpy as np
import tyro

from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.utils.launch_utils import instantiate_from_dict
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera
# from experiments import launch_camera_ROS_client
# from gello.data_utils.format_obs_queue import AsyncSaver
from gello.robots.xarm_robot import Rate
#from experiments.zmq_tactile_client import ZMQClientTactile

# from spatialmath import SE3
# from roboticstoolbox import DHRobot, RevoluteDH, Robot
# import roboticstoolbox as rtb

# def define_xarm6():
#     xarm = rtb.DHRobot([
#             rtb.RevoluteDH(a=0.0, alpha=-np.pi/2, d=.267, offset=0),           # Joint 1
#             rtb.RevoluteDH(a=.28948, alpha=0, d=0, offset=-1.384),          # Joint 2
#             rtb.RevoluteDH(a=.0775, alpha=-np.pi/2, d=0, offset= 1.38491),    # Joint 3
#             rtb.RevoluteDH(a=0.0, alpha=np.pi/2, d=.3425, offset=0),          # Joint 4
#             rtb.RevoluteDH(a=.0760, alpha=-np.pi/2, d=0, offset=0),            # Joint 5
#             rtb.RevoluteDH(a=0.0, alpha=0, d=.097, offset=0),                # Joint 6
#         ], name='xArm')
#     return xarm

# def compute_forward_kinematics(robot: DHRobot, q: np.ndarray) -> SE3:
#     q = np.asarray(q).flatten()
#     T = robot.fkine(q)  # returns SE3
#     return T.t


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    label: str = None 
    robot_port: int = 6001
    wrist_camera_port: int = 4000
    base_camera_port: int = 4001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 30
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9HDFUF-if00-port0"
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/test/"
    bimanual: bool = False
    verbose: bool = False
    use_sensor: bool = False
    config_index: int = 0

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)


def main(args):
    from gello.utils.control_utils import SaveInterface, run_control_loop

    # --- One-time setup ---
    #tactile_client = None
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        camera_clients = {}
        #if args.use_sensor:
         #   tactile_client = ZMQClientTactile(
           #     host=args.hostname,
           #     port=args.tactile_port,
           #     hz=float(args.tactile_hz),
           #     timeout_ms=200,
          #  )
          #  tactile_client.start()
          #  print("Checking tactile server connection...")
          #  time.sleep(1.0)
          #  snap, _, _ = tactile_client.get_latest()
          #  if not snap:
          #      raise RuntimeError(
          #          f"Tactile server not reachable at tcp://{args.hostname}:{args.tactile_port}. "
          #          f"Please run: python experiments/tactile_nodes.py"
          #      )

    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    # Load all configs from CSV once
    csv_path = Path(__file__).resolve().parent.parent / "scripts" / "init_target_configs_left.csv"
    configs = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding="utf-8")
    total = len(configs)
    print(f"Loaded {total} configs from {csv_path.name}. Starting from config {args.config_index}.")

    # --- Build agent (once, reused across configs) ---
    if args.bimanual:
        if args.agent == "gello":
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {"_target_": "gello.agents.gello_agent.GelloAgent", "port": left},
                "agent_right": {"_target_": "gello.agents.gello_agent.GelloAgent", "port": right},
            }
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {"_target_": "gello.agents.quest_agent.SingleArmQuestAgent", "robot_type": args.robot_type, "which_hand": "l"},
                "agent_right": {"_target_": "gello.agents.quest_agent.SingleArmQuestAgent", "robot_type": args.robot_type, "which_hand": "r"},
            }
        elif args.agent == "spacemouse":
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {"_target_": "gello.agents.spacemouse_agent.SpacemouseAgent", "robot_type": args.robot_type, "device_path": "/dev/hidraw0", "verbose": args.verbose},
                "agent_right": {"_target_": "gello.agents.spacemouse_agent.SpacemouseAgent", "robot_type": args.robot_type, "device_path": "/dev/hidraw1", "verbose": args.verbose, "invert_button": True},
            }
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")
        # bimanual reset
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        steps = min(int(np.abs(curr_joints - reset_joints).max() / 0.01), 100)
        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
        agent: Any = instantiate_from_dict(agent_cfg)
        run_control_loop(env, agent, None, use_colors=True)
        return

    # Single-arm agent setup
    if args.agent == "gello":
        gello_port = args.gello_port
        if gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"using port {gello_port}")
            else:
                raise ValueError("No gello port found, please specify one or plug in gello")
        agent_cfg = {
            "_target_": "gello.agents.gello_agent.GelloAgent",
            "port": gello_port,
            "start_joints": args.start_joints,
        }
    elif args.agent == "quest":
        agent_cfg = {"_target_": "gello.agents.quest_agent.SingleArmQuestAgent", "robot_type": args.robot_type, "which_hand": "l"}
    elif args.agent == "spacemouse":
        agent_cfg = {"_target_": "gello.agents.spacemouse_agent.SpacemouseAgent", "robot_type": args.robot_type, "verbose": args.verbose}
    elif args.agent in ("dummy", "none"):
        agent_cfg = {"_target_": "gello.agents.agent.DummyAgent", "num_dofs": robot_client.num_dofs()}
    elif args.agent == "policy":
        raise NotImplementedError("add your imitation policy here if there is one")
    else:
        raise ValueError("Invalid agent name")

    agent: Any = instantiate_from_dict(agent_cfg)

    # --- Per-config loop ---
    for config_idx in range(args.config_index, total):
        print(f"\n{'='*52}")
        print(f"  Config {config_idx} / {total - 1}")
        print(f"{'='*52}")

        cfg = configs[config_idx]
        init_joints = np.array([cfg['init_j1'], cfg['init_j2'], cfg['init_j3'],
                                 cfg['init_j4'], cfg['init_j5'], cfg['init_j6']])
        tgt_position = np.array([cfg['tgt_x'], cfg['tgt_y'], cfg['tgt_z']])
        tgt_joints = np.array([cfg['tgt_j1'], cfg['tgt_j2'], cfg['tgt_j3'],
                                cfg['tgt_j4'], cfg['tgt_j5'], cfg['tgt_j6']])
        #contact_joints = np.array([cfg['sensor_contact_j1'], cfg['sensor_contact_j2'], cfg['sensor_contact_j3'],
                                    #cfg['sensor_contact_j4'], cfg['sensor_contact_j5'], cfg['sensor_contact_j6']])
        #contact_orient_deg = float(cfg['contact_orient_x_deg'])
        #contact_type = str(cfg['contact_type']).strip()
        #type_label = {"S": "Small", "M": "Medium", "L": "Large"}.get(contact_type.upper(), contact_type)

        print(f"  init={np.round(init_joints, 3)}")
        print(f"  target_pos={np.round(tgt_position, 3)}")
        #print(f"  contact_type={type_label}  contact_orient_x={contact_orient_deg:.2f} deg")

        if not args.mock:
            robot_client.set_init_position(np.array([cfg['init_x'], cfg['init_y'], cfg['init_z']]))
            robot_client.set_target_position(tgt_position)
            #robot_client.set_sensor_contact_position(np.array([cfg['sensor_contact_x'], cfg['sensor_contact_y'], cfg['sensor_contact_z']]))

        if args.agent == "gello":
            curr_joints = env.get_obs()["joint_positions"]
            #gripper = curr_joints[-1:]  # preserve gripper across moves
            init_cmd = np.concatenate([init_joints])
            tgt_cmd = np.concatenate([tgt_joints])

            print("Moving to init position...")
            steps = max(50, min(int(np.abs(curr_joints - init_cmd).max() / 0.01), 200))
            for jnt in np.linspace(curr_joints, init_cmd, steps):
                env.step(jnt)
                time.sleep(0.001)

            #input("At init position. Press Enter to move to target...")

            #print("Moving to target to preview path...")
            #curr_joints = env.get_obs()["joint_positions"]
            #steps = max(50, min(int(np.abs(curr_joints - tgt_cmd).max() / 0.01), 200))
            #for jnt in np.linspace(curr_joints, tgt_cmd, steps):
                #env.step(jnt)
                #time.sleep(0.001)

            #input("At target position. Press Enter to move to contact location...")

            #contact_cmd = np.concatenate([contact_joints, gripper])
            #print("Moving to contact location...")
            #curr_joints = env.get_obs()["joint_positions"]
            #steps = max(50, min(int(np.abs(curr_joints - contact_cmd).max() / 0.01), 200))
            #for jnt in np.linspace(curr_joints, contact_cmd, steps):
                #env.step(jnt)
                #time.sleep(0.001)

            #print(f"\n  Object type  : {type_label} ({contact_type})")
            #print(f"  Contact orient (x): {contact_orient_deg:.2f} deg")

            #input("At contact location. Press Enter to move back to init...")

            #print("Moving back to init position...")
            #curr_joints = env.get_obs()["joint_positions"]
            #steps = max(50, min(int(np.abs(curr_joints - init_cmd).max() / 0.01), 200))
            #for jnt in np.linspace(curr_joints, init_cmd, steps):
            #    env.step(jnt)
            #    time.sleep(0.001)

            input("At init position. Press Enter to check gello offset (move gello until offset is near zero)...")

        # Live gello offset display — user moves gello to match robot, then presses Enter
        import threading
        stop_event = threading.Event()

        def _wait_enter():
            input("Press Enter when gello offset is near zero to collect data...")
            stop_event.set()

        enter_thread = threading.Thread(target=_wait_enter, daemon=True)
        enter_thread.start()

        print("Move gello to match robot position. Offset per joint:")
        while not stop_event.is_set():
            obs = env.get_obs()
            gello_pos = agent.act(obs)
            offset = gello_pos - obs["joint_positions"]
            max_offset = np.abs(offset).max()
            flag = "\033[92m[OK - ready to collect]\033[0m" if max_offset <= np.deg2rad(3) else "\033[91m[NOT OK]\033[0m"
            print(f"\r  offset={np.round(offset, 3)}  max={max_offset:.3f}  {flag}   ", end="", flush=True)
            time.sleep(0.1)
        print()

        save_interface = None
        if args.use_save_interface:
            label = f"{args.agent}_contact" if args.use_sensor else args.agent
            if args.use_sensor:
                tact_check = env.get_obs().get("tactile_data")
                if tact_check is None:
                    print("\033[93m[WARNING] Tactile sensor not yet calibrated — tactile_data will be None until calibration completes.\033[0m")
                else:
                    print(f"\033[92m[OK] Tactile data ready, shape={tact_check.shape}\033[0m")
            save_interface = SaveInterface(data_dir=args.data_dir, agent_name=label, expand_user=True)

        run_control_loop(env, agent, save_interface, use_colors=True)
        print(f"\nConfig {config_idx} complete. Next config index to resume from: {config_idx + 1}")
        input(f"Press Enter to move to config {config_idx + 1} init position...")


if __name__ == "__main__":
    main(tyro.cli(Args))
