/*
use std::ffi::c_void;

use ndarray::Array1;

use crate::environement::environment_traits::{BaseEnv, Environment};
use crate::utils::lib_utils::LIB;

pub const NUM_ACTIONS: usize = 4;
pub const NUM_STATES: usize = 49;
pub const NUM_REWARDS: usize = 4;
/// The `SecretEnv1` struct represents an environment that interacts with an external library to perform various operations.
/// This struct implements the `Environment` trait, allowing it to be used in reinforcement learning algorithms.
///
/// # Fields
///
/// - `env`: A pointer to the environment object managed by the external library.
///
#[derive(Clone, Debug)]
pub struct SecretEnv1 {
    pub env: *mut c_void,
}

impl Default for SecretEnv1 {
    fn default() -> Self {
        let secret_env_1_new: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> =
            unsafe { LIB.get(b"secret_env_1_new") }
                .expect("Failed to load function `secret_env_1_new`");
        unsafe {
            let env = secret_env_1_new();
            SecretEnv1 { env }
        }
    }
}

impl BaseEnv for SecretEnv1 {
    fn is_terminal(&self) -> bool {
        let secret_env_1_is_game_over: libloading::Symbol<
            unsafe extern "C" fn(*const c_void) -> bool,
        > = unsafe { LIB.get(b"secret_env_1_is_game_over") }
            .expect("Failed to load function `secret_env_1_is_game_over`");
        return unsafe { secret_env_1_is_game_over(self.env) };
    }
    fn score(&self) -> f32 {
        unsafe {
            let secret_env_1_score: libloading::Symbol<unsafe extern "C" fn(*const c_void) -> f32> =
                LIB.get(b"secret_env_1_score")
                    .expect("Failed to load function `secret_env_1_score`");
            secret_env_1_score(self.env)
        }
    }
    fn reset(&mut self) {
        let secret_env_1_reset: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            unsafe { LIB.get(b"secret_env_1_reset") }
                .expect("Failed to load function `secret_env_1_reset`");
        unsafe { secret_env_1_reset(self.env) };
    }
}

impl Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS> for SecretEnv1 {
    fn state_id(&self) -> usize {
        let secret_env_1_state_id: libloading::Symbol<
            unsafe extern "C" fn(*const c_void) -> usize,
        > = unsafe { LIB.get(b"secret_env_1_state_id") }
            .expect("Failed to load function `secret_env_1_state_id`");
        return unsafe { secret_env_1_state_id(self.env) };
    }

    fn num_states() -> usize {
        let secret_env_1_num_states: libloading::Symbol<unsafe extern "C" fn() -> usize> =
            unsafe { LIB.get(b"secret_env_1_num_states") }
                .expect("Failed to load function `secret_env_1_num_states`");
        return unsafe { secret_env_1_num_states() };
    }

    fn num_actions() -> usize {
        let secret_env_1_num_actions: libloading::Symbol<unsafe extern "C" fn() -> usize> =
            unsafe { LIB.get(b"secret_env_1_num_actions") }
                .expect("Failed to load function `secret_env_1_num_actions`");
        return unsafe { secret_env_1_num_actions() };
    }

    fn num_rewards() -> usize {
        let secret_env_1_num_rewards: libloading::Symbol<unsafe extern "C" fn() -> usize> =
            unsafe { LIB.get(b"secret_env_1_num_rewards") }
                .expect("Failed to load function `secret_env_1_num_rewards`");
        return unsafe { secret_env_1_num_rewards() };
    }

    fn get_reward(i: usize) -> f32 {
        let secret_env_1_reward: libloading::Symbol<unsafe extern "C" fn(usize) -> f32> =
            unsafe { LIB.get(b"secret_env_1_reward") }
                .expect("Failed to load function `secret_env_1_reward`");
        return unsafe { secret_env_1_reward(i) };
    }


    fn reset_random_state(&mut self, _seed: u64) {
        unsafe {
            let secret_env_1_from_random_state: libloading::Symbol<
                unsafe extern "C" fn() -> *mut c_void,
            > = LIB
                .get(b"secret_env_1_from_random_state")
                .expect("Failed to load function `secret_env_1_from_random_state`");
            let env = secret_env_1_from_random_state();
            self.env = env
        }
    }

    fn from_random_state() -> Self {
        unsafe {
            let secret_env_1_from_random_state: libloading::Symbol<
                unsafe extern "C" fn() -> *mut c_void,
            > = LIB
                .get(b"secret_env_1_from_random_state")
                .expect("Failed to load function `secret_env_1_from_random_state`");
            let env = secret_env_1_from_random_state();
            SecretEnv1 { env }
        }
    }

    fn available_actions_ids(&self) -> Array1<usize> {
        unsafe {
            let mut aa = Vec::new();
            let secret_env_1_available_actions: libloading::Symbol<
                unsafe extern "C" fn(*const c_void) -> *const usize,
            > = LIB
                .get(b"secret_env_1_available_actions")
                .expect("Failed to load function `secret_env_1_available_actions`");
            let actions = secret_env_1_available_actions(self.env);

            let secret_env_1_available_actions_len: libloading::Symbol<
                unsafe extern "C" fn(*const c_void) -> usize,
            > = LIB
                .get(b"secret_env_1_available_actions_len")
                .expect("Failed to load function `secret_env_1_available_actions_len`");

            // show all available actions
            for i in 0..secret_env_1_available_actions_len(self.env) {
                aa.push(*actions.add(i));
            }
            Array1::from_vec(aa)
        }
    }

    fn available_action_delete(&self) {
        todo!()
    }


    /*
    fn is_forbidden(&self, action: usize) -> bool {
        let secret_env_1_is_forbidden: libloading::Symbol<
            unsafe extern "C" fn(*const c_void, usize) -> bool,
        > = unsafe { LIB.get(b"secret_env_1_is_forbidden") }
            .expect("Failed to load function `secret_env_1_is_forbidden`");
        return unsafe { secret_env_1_is_forbidden(self.env, action) };
    }
     */

    fn step(&mut self, action: usize) {
        unsafe {
            let secret_env_1_step: libloading::Symbol<unsafe extern "C" fn(*mut c_void, usize)> =
                LIB.get(b"secret_env_1_step")
                    .expect("Failed to load function `secret_env_1_step`");
            secret_env_1_step(self.env, action);
        }
    }

    fn delete(&mut self) {
        unsafe {
            let secret_env_1_delete: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = LIB
                .get(b"secret_env_1_delete")
                .expect("Failed to load function `secret_env_1_delete`");
            secret_env_1_delete(self.env)
        }
    }


    fn display(&self) {
        unsafe {
            let secret_env_1_display: libloading::Symbol<unsafe extern "C" fn(*const c_void)> = LIB
                .get(b"secret_env_1_display")
                .expect("Failed to load function `secret_env_1_display`");
            secret_env_1_display(self.env)
        }
    }


    fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        let secret_env_1_transition_probability: libloading::Symbol<
            unsafe extern "C" fn(usize, usize, usize, usize) -> f32,
        > = unsafe { LIB.get(b"secret_env_1_transition_probability") }
            .expect("Failed to load function `secret_env_1_transition_probability`");
        return unsafe { secret_env_1_transition_probability(s, a, s_p, r) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let secret0 = SecretEnv1::default();
        dbg!(secret0.env);
        dbg!(SecretEnv1::num_states());
        assert_eq!(SecretEnv1::num_states(), 8192);
        dbg!(SecretEnv1::num_actions());
        assert_eq!(SecretEnv1::num_actions(), 3);
        dbg!(SecretEnv1::num_rewards());
        assert_eq!(SecretEnv1::num_rewards(), 3);
        for i in 0..SecretEnv1::num_rewards() {
            dbg!(SecretEnv1::get_reward(i));
        }
        assert_eq!(SecretEnv1::get_reward(0), -1.0);
        assert_eq!(SecretEnv1::get_reward(1), 0.0);
        assert_eq!(SecretEnv1::get_reward(2), 1.0);

        let mut env = SecretEnv1::default();
        dbg!(env.state_id());

        assert_eq!(
            SecretEnv1::build_transition_probability(0, 0, 0, 0),
            0.0
        );

        let secret_env_1_state_id: libloading::Symbol<
            unsafe extern "C" fn(*const c_void) -> usize,
        > = unsafe { LIB.get(b"secret_env_1_state_id") }
            .expect("Failed to load function `secret_env_1_state_id`");
        unsafe {
            dbg!(secret_env_1_state_id(env.env));
        }

        unsafe {
            let secret_env_1_new: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> = LIB
                .get(b"secret_env_1_new")
                .expect("Failed to load function `secret_env_1_new`");
            let env2_p = secret_env_1_new();

            let env3 = SecretEnv1::default();
            dbg!(env3.env);
            dbg!(env3.state_id());
            dbg!(secret_env_1_state_id(env3.env));

            let env3_p = env3.env;

            dbg!(env2_p);
            dbg!(env3_p);
            println!("{:?}", env3_p);
            println!("{:?}", env2_p);

            dbg!(secret_env_1_state_id(env2_p));
        }

        assert_eq!(env.state_id(), 0);
        assert_eq!(env.is_terminal(), false);
    }
}


 */