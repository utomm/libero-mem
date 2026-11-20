from __future__ import annotations

import collections
import csv
import dataclasses
import json
import logging
import math
import pathlib
import re
from typing import Optional, Tuple

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy
from PIL import Image
import robosuite.utils.transform_utils as T
import torch
import tqdm
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
TARGET_SIZE = (224, 224)
EPS = 1e-6
TASK_MAX_STEPS = {
    1: 200,
    2: 290,
    3: 320,
    4: 370,
    5: 450,
    6: 655,
    7: 480,
    8: 675,
    9: 390,
    10: 390,
}
INIT_SOURCE_TASK = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 7, 8: 8, 9: 9, 10: 9}


def process_image(image: np.ndarray) -> np.ndarray:
    """Flip and resize a simulator RGB image for EVA inference.

    Args:
        image: RGB image with shape [H, W, 3].

    Returns:
        RGB uint8 image with shape [224, 224, 3].
    """
    image = np.ascontiguousarray(image[::-1])
    return np.asarray(Image.fromarray(image).resize(TARGET_SIZE, Image.Resampling.BILINEAR))


def mat_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:3, 0], matrix[:3, 1]])


def rot6d_to_mat(rotation: np.ndarray) -> np.ndarray:
    first, second = rotation[:3], rotation[3:6]
    basis1 = first / (np.linalg.norm(first) + EPS)
    basis2 = second - np.dot(basis1, second) * basis1
    basis2 = basis2 / (np.linalg.norm(basis2) + EPS)
    return np.stack([basis1, basis2, np.cross(basis1, basis2)], axis=1)


def world_to_base(position, orientation, base_position, base_orientation):
    return base_orientation.T @ (position - base_position), base_orientation.T @ orientation


def base_to_world(position, orientation, base_position, base_orientation):
    return base_orientation @ position + base_position, base_orientation @ orientation


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] ** 2)
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(quaternion[3]) / denominator


def get_state_from_controller(controller, gripper_qpos, base_position, base_orientation):
    position, orientation = world_to_base(
        controller.ee_pos.copy(), controller.ee_ori_mat.copy(), base_position, base_orientation
    )
    return np.concatenate([position, mat_to_rot6d(orientation), gripper_qpos]).astype(np.float32)


def action_to_env_command(action, base_position, base_orientation):
    orientation = rot6d_to_mat(action[3:9])
    position, orientation = base_to_world(action[:3], orientation, base_position, base_orientation)
    return np.concatenate([position, quat_to_axis_angle(T.mat2quat(orientation)), [action[-1]]])


def _task_number(task_name: str) -> int:
    match = re.search(r"KITCHEN_SCENE1_(\d+)_", task_name)
    if match is None:
        raise ValueError(f"Cannot parse LIBERO-Mem task number from {task_name!r}")
    return int(match.group(1))


def _load_initial_states(task_name: str) -> np.ndarray:
    task_number = _task_number(task_name)
    init_root = pathlib.Path(__file__).resolve().parent / "scripts" / "__init_data"
    matches = sorted(init_root.glob(f"KITCHEN_SCENE1_{INIT_SOURCE_TASK[task_number]}_*.pruned_init"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one init-state file for task {task_number}, found {matches}")
    return np.asarray(torch.load(matches[0], map_location="cpu"))


def _predicate_text(predicate) -> str:
    return f"{predicate[0]}({', '.join(predicate[1:])})"


def _subtask_status(problem) -> tuple[list[dict], str]:
    sequences = problem.get_all_goals()
    completed_predicates = [subgoal[1] for subgoal in problem._satisfied_subgoals]
    completed_keys = [tuple(predicate) for predicate in completed_predicates]
    candidates = []
    for sequence in sequences:
        sequence_keys = [tuple(subtask[0]) for subtask in sequence]
        if sequence_keys[: len(completed_keys)] == completed_keys:
            candidates.append(sequence)
    candidates = candidates or sequences

    status = []
    total = problem.get_goal_sequence_len()
    for index in range(total):
        if index < len(completed_predicates):
            descriptions = [_predicate_text(completed_predicates[index])]
        else:
            descriptions = sorted({_predicate_text(sequence[index][0]) for sequence in candidates})
        status.append(
            {
                "index": index + 1,
                "description": " | ".join(descriptions),
                "completed": index < len(completed_predicates),
            }
        )

    if len(completed_predicates) < total:
        stuck = status[len(completed_predicates)]["description"]
    elif problem._overshot:
        stuck = "overshot_after_completion"
    elif not problem._check_success():
        stuck = "final_subtask_not_currently_satisfied"
    else:
        stuck = ""
    return status, stuck


def _stabilize_initial_state(env, problem, initial_state: np.ndarray, num_steps: int):
    """Set an initial state and settle movable objects from the bottom up.

    Args:
        env: LIBERO off-screen environment wrapper.
        problem: Underlying BDDL environment.
        initial_state: Flattened MuJoCo state loaded from the task init file.
        num_steps: Number of simulator steps for each settling phase.

    Returns:
        The observation after stabilization.
    """
    obs = env.set_init_state(initial_state)
    relations = [
        state
        for state in problem.parsed_problem["initial_state"]
        if len(state) == 3
        and state[0].lower() == "on"
        and state[1] in problem.objects_dict
        and state[2] in problem.objects_dict
    ]
    if not relations:
        for _ in range(num_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        return obs

    held_objects = {}
    initial_axes = {}
    support_positions = {}
    for _, object_name, support_name in relations:
        object_model = problem.objects_dict[object_name]
        support_model = problem.objects_dict[support_name]
        object_joint = object_model.joints[0]
        support_joint = support_model.joints[0]
        held_objects[object_name] = (object_joint, np.asarray(problem.sim.data.get_joint_qpos(object_joint)).copy())
        support_positions[support_name] = np.asarray(problem.sim.data.get_joint_qpos(support_joint))[:3].copy()
        body_id = problem.sim.model.body_name2id(object_model.root_body)
        initial_axes[object_name] = np.asarray(problem.sim.data.body_xmat[body_id]).reshape(3, 3)[:, 2].copy()

    for object_joint, object_qpos in held_objects.values():
        held_qpos = object_qpos.copy()
        held_qpos[2] += 1.0
        problem.sim.data.set_joint_qpos(object_joint, held_qpos)
        problem.sim.data.set_joint_qvel(object_joint, np.zeros(6))
    problem.sim.forward()

    for _ in range(num_steps):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        for object_joint, object_qpos in held_objects.values():
            held_qpos = object_qpos.copy()
            held_qpos[2] += 1.0
            problem.sim.data.set_joint_qpos(object_joint, held_qpos)
            problem.sim.data.set_joint_qvel(object_joint, np.zeros(6))
        problem.sim.forward()

    for _, object_name, support_name in relations:
        object_joint, object_qpos = held_objects[object_name]
        support_joint = problem.objects_dict[support_name].joints[0]
        support_position = np.asarray(problem.sim.data.get_joint_qpos(support_joint))[:3]
        restored_qpos = object_qpos.copy()
        restored_qpos[:3] += support_position - support_positions[support_name]
        for _ in range(250):
            problem.sim.data.set_joint_qpos(object_joint, restored_qpos)
            problem.sim.data.set_joint_qvel(object_joint, np.zeros(6))
            problem.sim.forward()
            if problem._eval_predicate(["on", object_name, support_name]):
                break
            restored_qpos[2] -= 0.002
        else:
            raise RuntimeError(f"Could not place {object_name} on {support_name} during initialization")
    problem.sim.forward()

    for _ in range(num_steps):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    for relation in relations:
        _, object_name, support_name = relation
        object_model = problem.objects_dict[object_name]
        body_id = problem.sim.model.body_name2id(object_model.root_body)
        final_axis = np.asarray(problem.sim.data.body_xmat[body_id]).reshape(3, 3)[:, 2]
        dot = np.clip(np.dot(initial_axes[object_name], final_axis), -1.0, 1.0)
        tilt_degrees = math.degrees(math.acos(dot))
        qvel = np.asarray(problem.sim.data.get_joint_qvel(object_model.joints[0]))
        linear_speed = np.linalg.norm(qvel[:3])
        angular_speed = np.linalg.norm(qvel[3:])
        predicate_satisfied = bool(problem._eval_predicate(relation))
        logging.info(
            "Stabilized %s on %s: predicate=%s tilt=%.2f degrees linear_speed=%.4f angular_speed=%.4f",
            object_name,
            support_name,
            predicate_satisfied,
            tilt_degrees,
            linear_speed,
            angular_speed,
        )
        if not predicate_satisfied or tilt_degrees > 30.0 or linear_speed > 0.05 or angular_speed > 0.25:
            raise RuntimeError(
                f"Initial state validation failed for {object_name} on {support_name}: "
                f"predicate={predicate_satisfied}, tilt={tilt_degrees:.2f} degrees, "
                f"linear_speed={linear_speed:.4f}, angular_speed={angular_speed:.4f}"
            )
    return obs


def get_libero_env(task, resolution: int, seed: int):
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=False,
        camera_segmentations=None,
    )
    env.seed(seed)
    instruction_value = env.env.parsed_problem["language_instruction"]
    instruction = (
        " ".join(instruction_value).strip()
        if isinstance(instruction_value, list)
        else instruction_value.strip()
    )
    return env, instruction


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 9000
    replan_steps: int = 8
    task_suite_name: str = "libero_mem"
    num_steps_wait: int = 20
    num_trials_per_task: int = 20
    init_indices: Optional[Tuple[int, ...]] = None
    stabilize_init: bool = True
    discretize_gripper: bool = True
    gripper_threshold: float = 0.0
    video_out_path: str = "data/libero_mem/videos"
    seed: int = 7
    rank: int = 0
    experiment_root: str = ""
    no_video: bool = False
    max_steps: int = 0


def _output_root(args: Args) -> pathlib.Path:
    return pathlib.Path(args.experiment_root) if args.experiment_root else pathlib.Path(args.video_out_path).parent


def eval_libero(args: Args) -> None:
    """Evaluate EVA on LIBERO-Mem with deterministic initial states and subtask metrics."""
    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    output_root = _output_root(args)
    results_path = output_root / "results" / f"{args.rank}.csv"
    end_path = output_root / "end" / f"{args.rank}.end"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    end_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "task_suite_name",
        "task_id",
        "episode_idx",
        "init_index",
        "seed",
        "success",
        "completion",
        "completed_subtasks",
        "total_subtasks",
        "overshot",
        "stuck_subtask",
        "subtask_status",
        "error",
    ]
    with results_path.open("w", newline="") as results_file:
        csv.DictWriter(results_file, fieldnames=fieldnames).writeheader()

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    total_episodes = 0
    total_successes = 0
    completion_sum = 0.0

    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        task_number = _task_number(task.name)
        initial_states = _load_initial_states(task.name)
        if args.init_indices is not None:
            init_indices = np.asarray(args.init_indices, dtype=np.int64)
            if np.any(init_indices < 0) or np.any(init_indices >= len(initial_states)):
                raise ValueError(f"Init indices must be in [0, {len(initial_states) - 1}]: {init_indices.tolist()}")
        else:
            if args.num_trials_per_task > len(initial_states):
                raise ValueError(f"Requested {args.num_trials_per_task} trials from {len(initial_states)} init states")
            init_rng = np.random.RandomState(args.seed + task_number * 1009)
            init_indices = init_rng.permutation(len(initial_states))[: args.num_trials_per_task]
        env, instruction = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        problem = env.env
        max_steps = args.max_steps or TASK_MAX_STEPS[task_number]

        for episode_idx, init_index in enumerate(init_indices):
            logging.info(
                "Task %d episode %d: instruction=%r init_index=%d seed=%d",
                task_number,
                episode_idx,
                instruction,
                init_index,
                args.seed,
            )
            error = ""
            replay_images = []
            env.reset()
            for robot in problem.robots:
                robot.controller.use_delta = True
            if args.stabilize_init:
                obs = _stabilize_initial_state(env, problem, initial_states[init_index], args.num_steps_wait)
            else:
                obs = env.set_init_state(initial_states[init_index])
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            problem.reset_subgoal_progress()
            policy.reset()

            base_position = problem.robots[0].base_pos.copy()
            base_orientation = T.quat2mat(problem.robots[0].base_ori)
            controller = problem.robots[0].controller
            action_plan = collections.deque()

            for robot in problem.robots:
                robot.controller.use_delta = False

            previous_completed = 0
            for step in range(max_steps):
                try:
                    image = process_image(obs["agentview_image"])
                    wrist_image = process_image(obs["robot0_eye_in_hand_image"])
                    replay_images.append(image)
                    if not action_plan:
                        state = get_state_from_controller(
                            controller,
                            obs["robot0_gripper_qpos"],
                            base_position,
                            base_orientation,
                        )
                        request = {
                            "observation/image": image,
                            "observation/wrist_image": wrist_image,
                            "observation/state": state,
                            "prompt": instruction,
                            "task": instruction,
                            "observation.images.image": image,
                            "observation.images.wrist_image": wrist_image,
                            "observation.state": state,
                        }
                        response = policy.infer(request)
                        actions = response.get("actions")
                        if actions is None or len(actions) < args.replan_steps:
                            raise RuntimeError("Policy returned an incomplete action chunk")
                        action_plan.extend(actions[: args.replan_steps])

                    action = np.asarray(action_plan.popleft()).copy()
                    if args.discretize_gripper:
                        action[-1] = 1.0 if action[-1] > args.gripper_threshold else -1.0
                    command = action_to_env_command(action, base_position, base_orientation)
                    obs, _, _, _ = env.step(command.tolist())
                    problem._check_success(inc=True)
                    completed = len(problem._satisfied_subgoals)
                    if completed > previous_completed:
                        status, _ = _subtask_status(problem)
                        logging.info(
                            "Task %d episode %d step %d completed subtask %d/%d: %s",
                            task_number,
                            episode_idx,
                            step,
                            completed,
                            problem.get_goal_sequence_len(),
                            status[completed - 1]["description"],
                        )
                        previous_completed = completed
                except Exception as exception:
                    logging.exception("Task %d episode %d failed during rollout", task_number, episode_idx)
                    error = f"{type(exception).__name__}: {exception}"
                    break

            subtask_status, stuck_subtask = _subtask_status(problem)
            completed_subtasks = len(problem._satisfied_subgoals)
            total_subtasks = problem.get_goal_sequence_len()
            completion = completed_subtasks / total_subtasks
            success = bool(problem._check_success()) and not problem._overshot and not error
            logging.info(
                "Task %d episode %d result: success=%s completion=%d/%d overshot=%s stuck=%s",
                task_number,
                episode_idx,
                success,
                completed_subtasks,
                total_subtasks,
                problem._overshot,
                stuck_subtask or "none",
            )

            if not args.no_video and replay_images:
                suffix = "success" if success else "failure"
                video_path = pathlib.Path(args.video_out_path) / (
                    f"task{task_number}_episode{episode_idx}_init{init_index}_{suffix}.mp4"
                )
                imageio.mimwrite(video_path, replay_images, fps=10)

            row = {
                "rank": args.rank,
                "task_suite_name": args.task_suite_name,
                "task_id": task_id,
                "episode_idx": episode_idx,
                "init_index": int(init_index),
                "seed": args.seed,
                "success": int(success),
                "completion": completion,
                "completed_subtasks": completed_subtasks,
                "total_subtasks": total_subtasks,
                "overshot": int(problem._overshot),
                "stuck_subtask": stuck_subtask,
                "subtask_status": json.dumps(subtask_status),
                "error": error,
            }
            with results_path.open("a", newline="") as results_file:
                csv.DictWriter(results_file, fieldnames=fieldnames).writerow(row)
            total_episodes += 1
            total_successes += int(success)
            completion_sum += completion

        env.close()

    end_path.write_text("")
    if total_episodes:
        logging.info("Total success rate: %.4f", total_successes / total_episodes)
        logging.info("Total completion rate: %.4f", completion_sum / total_episodes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
