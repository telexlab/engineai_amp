from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg 
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from engineai_lab.tasks.velocity import mdp
from engineai_lab.robots.pm01 import PM01_CFG, PM_WAIST_DFS_JOINT_NAMES, PM01_DFS_JOINT_ORDER_ASSET_CFG

import isaaclab.terrains as terrain_gen
import math
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from engineai_lab.robots.actuator import DelayedImplicitActuatorCfg


ACTUATOR_DELAY_RANGE = (2, 8)
def _build_delayed_actuators():
    delayed_actuators = {}
    for name, cfg in PM01_CFG.actuators.items():
        delayed_actuators[name] = DelayedImplicitActuatorCfg(
            joint_names_expr=cfg.joint_names_expr,
            effort_limit=cfg.effort_limit,
            effort_limit_sim=cfg.effort_limit_sim,
            velocity_limit=cfg.velocity_limit,
            velocity_limit_sim=cfg.velocity_limit_sim,
            stiffness=cfg.stiffness,
            damping=cfg.damping,
            armature=cfg.armature,
            friction=cfg.friction,
            dynamic_friction=cfg.dynamic_friction,
            viscous_friction=cfg.viscous_friction,
            min_delay=ACTUATOR_DELAY_RANGE[0],
            max_delay=ACTUATOR_DELAY_RANGE[1],
        )
    return delayed_actuators


terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            border_width=25.0,
            num_rows=10,
            num_cols=20,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            color_scheme="height",
            slope_threshold=0.75,
            sub_terrains={
                "flat": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.4,
                    slope_range=(0.0, 0.0),
                    platform_width=8.0,
                ),
                "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1,
                    slope_range=(0.0, math.radians(5)),
                    platform_width=2.0,
                ),
                "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1,
                    slope_range=(0.0, math.radians(5)),
                    platform_width=2.0,
                ),
                "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.2,
                    obstacle_width_range=(1.0, 2.0),
                    obstacle_height_range=(0.01, 0.1),
                    num_obstacles=15,
                    platform_width=3.0,
                ),
                "rough_terrain": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),
                    noise_step=0.005,
                    downsampled_scale=0.15,
                ),
            },
        )

@configclass
class PM01SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = PM01_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators=_build_delayed_actuators(),
    )
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LINK_BASE",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )



@configclass
class PM01Rewards:
    """Reward terms for the MDP."""


    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "sigma": 5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=2.5, 
        params={"command_name": "base_velocity", "sigma": 5}
    )

    base_orientation = RewTerm(
        func=mdp.base_orientation,
        weight=1.0
    )

    base_height = RewTerm(
        func=mdp.base_height_tracking,
        weight=0.4,
        params={"target_height": 0.82}
    )
    
    foot_position = RewTerm(
        func=mdp.feet_position,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "ankle_distance": 0.22,
            "base_height_target": 0.82,
        },
    )
    
    feet_orientation = RewTerm(
        func=mdp.feet_orientation,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
        },
    )
    
    waist_pos = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.3,
        params={
            "asset_cfg":  SceneEntityCfg("robot", joint_names=["J12_WAIST_YAW"]),
            "scale": 3.0,
            "tolerance": 0.0
        },
    )
    
    leg_joint_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HIP_ROLL_.*", ".*_HIP_YAW_.*", ".*_ANKLE_ROLL_.*"]),
                "scale": 3.0},
    )
    
    arm_pitch_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*SHOULDER_PITCH.*", ".*ELBOW_PITCH.*"]),
                "scale": 3.0},
    )
    
    arm_roll_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*SHOULDER_ROLL.*"]),
                "scale": 3.0},
    )
    
    arm_yaw_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*SHOULDER_YAW.*", ".*ELBOW_YAW.*"]),
                "scale": 10.0},
    )
    
    feet_contact = RewTerm(
        func=mdp.feet_contact_fixed,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "force_threshold": 5.0,
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "threshold": 0.5,
        },
    )
    
    # stand_still = RewTerm(
    #     func=mdp.stand_still_joint_deviation_l1,
    #     weight=-1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.1
    #     }
    # )
    
    feet_air_time_dense = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "threshold": 0.5,
        },        
    )

    foot_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "tangential_threshold": 2.0,
            "normal_threshold": 1.0,
        },
    )
    
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    
    energy_cost = RewTerm(
        func=mdp.energy_cost_with_curriculum,
        weight=-0.004,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "start_scale": 0.1,
                "power": 0.8,
                "interval_epochs": 200*24,
                },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
        },
    )

    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_rate = RewTerm(
        func=mdp.action_rate_with_curriculum,
        weight=-0.06,
        params={"start_scale": 0.1,
                "power": 0.8,
                "interval_epochs": 200*24
                },
    )

    action_smoothness = RewTerm(
        func=mdp.action_smoothness_with_curriculum,
        weight=-0.04,
        params={"start_scale": 0.1,
                "power": 0.8,
                "interval_epochs": 200*24
                },
    )
    
    dof_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)





@configclass
class PM01Termination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_BASE", "LINK_KNEE_PITCH.*", ".*SHOULDER.*", ".*ELBOW.*"]), "threshold": 1.0},
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",
                                           use_default_offset=True, 
                                           preserve_order=True,
                                           joint_names=PM_WAIST_DFS_JOINT_NAMES,
                                           scale = {".*_HIP_PITCH_.*" : 0.5,
                                                    ".*_HIP_ROLL_.*" : 0.2,
                                                    ".*_HIP_YAW_.*" : 0.2,
                                                    ".*_KNEE_PITCH_.*" : 0.5,
                                                    ".*_ANKLE_PITCH_.*" : 0.5,
                                                    ".*_ANKLE_ROLL_.*" : 0.2,
                                                    ".*WAIST_YAW.*" : 0.2,
                                                    ".*_SHOULDER_PITCH_.*" : 0.2,
                                                    ".*_SHOULDER_ROLL_.*" : 0.2,
                                                    ".*_SHOULDER_YAW_.*" : 0.2,
                                                    ".*_ELBOW_PITCH_.*" : 0.2,
                                                    ".*_ELBOW_YAW_.*" : 0.2
                                                }
                                           )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        
        actions = ObsTerm(func=mdp.last_action,
                          history_length=15)        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, 
                               noise=Unoise(n_min=-0.2, n_max=0.2),
                            history_length=15)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=15
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, 
                                    params={"command_name": "base_velocity"})        
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        
        actions = ObsTerm(func=mdp.last_action,
                          history_length=15)        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, 
                               noise=Unoise(n_min=-0.2, n_max=0.2),
                            history_length=15)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=15
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, 
                                    params={"command_name": "base_velocity"})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class PM01Commands:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.5, 7.5),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

@configclass
class PM01EventCfg:
    """PM01-specific randomizations."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    # rand_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mass_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #     },
    # )    

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (-0.5, 0.5)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class PM01FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration directly extending the base RL env config."""

    scene: PM01SceneCfg = PM01SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: PM01Commands = PM01Commands()
    rewards: PM01Rewards = PM01Rewards()
    terminations: PM01Termination = PM01Termination()
    events: PM01EventCfg = PM01EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Apply sim wiring and curriculum toggles."""
        # simulation settings
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = 0.005
            

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
