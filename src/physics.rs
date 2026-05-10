use rapier3d::prelude::*;

pub struct Physics {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    pub box_handle: RigidBodyHandle,
}

impl Physics {
    pub fn new() -> Self {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();

        colliders.insert(
            ColliderBuilder::cuboid(5.0, 0.1, 5.0)
                .translation(vector![0.0, -0.1, 0.0])
                .build(),
        );

        let box_handle = bodies.insert(
            RigidBodyBuilder::dynamic().translation(vector![0.0, 4.0, 0.0]).build(),
        );
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(0.5, 0.5, 0.5).build(),
            box_handle,
            &mut bodies,
        );

        Self {
            bodies,
            colliders,
            box_handle,
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
        }
    }

    pub fn step(&mut self) {
        self.pipeline.step(
            &vector![0.0, -9.81, 0.0],
            &IntegrationParameters::default(),
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );
    }

    pub fn box_matrix(&self) -> [[f32; 4]; 4] {
        let m = self.bodies[self.box_handle].position().to_homogeneous();
        let s = m.as_slice();
        [
            [s[0], s[1], s[2], s[3]],
            [s[4], s[5], s[6], s[7]],
            [s[8], s[9], s[10], s[11]],
            [s[12], s[13], s[14], s[15]],
        ]
    }
}
