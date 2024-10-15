use std::ffi::c_void;

use ndarray::Array1;

use crate::environement::environment_traits::Environment;
use crate::utils::lib_utils::LIB;

pub const NUM_ACTIONS: usize = 4;
pub const NUM_STATES: usize = 49;
pub const NUM_REWARDS: usize = 4;

/// The `SecretEnv2` struct represents an environment that interacts with an external library to perform various operations.
/// This struct implements the `Environment` trait, allowing it to be used in reinforcement learning algorithms.
///
/// # Fields
///
/// - `env`: A pointer to the environment object managed by the external library.
///
#[derive(Clone, Debug)]
pub struct SecretEnv2 {
    pub env: *mut c_void,
}

impl Default for SecretEnv2 {
    fn default() -> Self {
        let secret_env_2_new: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> =
            unsafe { LIB.get(b"secret_env_2_new") }
                .expect("Failed to load function `secret_env_2_new`");
        unsafe {
            let env = secret_env_2_new();
            SecretEnv2 { env }
        }
    }
}

impl Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS> for SecretEnv2 {

    fn state_id(&self) -> usize {
        let secret_env_2_state_id: libloading::Symbol<
            unsafe extern "C" fn(*const c_void) -> usize,
        > = unsafe { LIB.get(b"secret_env_2_state_id") }
            .expect("Failed to load function `secret_env_2_state_id`");
        return unsafe { secret_env_2_state_id(self.env) };
    }

    fn from_random_state() -> Self {
        unsafe {
            let secret_env_2_from_random_state: libloading::Symbol<
                unsafe extern "C" fn() -> *mut c_void,
            > = LIB
                .get(b"secret_env_2_from_random_state")
                .expect("Failed to load function `secret_env_2_from_random_state`");
            let env = secret_env_2_from_random_state();
            SecretEnv2 { env }
        }
    }

    fn reset(&mut self) {
        let secret_env_2_reset: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            unsafe { LIB.get(b"secret_env_2_reset") }
                .expect("Failed to load function `secret_env_2_reset`");
        unsafe { secret_env_2_reset(self.env) };
    }

    fn num_states() -> usize {
        let secret_env_2_num_states: libloading::Symbol<unsafe extern "C" fn() -> usize> =
            unsafe { LIB.get(b"secret_env_2_num_states") }
                .expect("Failed to load function `secret_env_2_num_states`");
        return unsafe { secret_env_2_num_states() };
    }

    fn num_actions() -> usize {
        let secret_env_2_num_actions: libloading::Symbol<unsafe extern "C" fn() -> usize> =
            unsafe { LIB.get(b"secret_env_2_num_actions") }
                .expect("Failed to load function `secret_env_2_num_actions`");
        return unsafe { secret_env_2_num_actions() };
    }


    fn num_rewards() -> usize {
        let secret_env_2_num_rewards: libloading::Symbol<unsafe extern "C" fn() -> usize> =
            unsafe { LIB.get(b"secret_env_2_num_rewards") }
                .expect("Failed to load function `secret_env_2_num_rewards`");
        return unsafe { secret_env_2_num_rewards() };
    }

    fn get_reward(i: usize) -> f32 {
        let secret_env_2_reward: libloading::Symbol<unsafe extern "C" fn(usize) -> f32> =
            unsafe { LIB.get(b"secret_env_2_reward") }
                .expect("Failed to load function `secret_env_2_reward`");
        return unsafe { secret_env_2_reward(i) };
    }

    fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        let secret_env_2_transition_probability: libloading::Symbol<
            unsafe extern "C" fn(usize, usize, usize, usize) -> f32,
        > = unsafe { LIB.get(b"secret_env_2_transition_probability") }
            .expect("Failed to load function `secret_env_2_transition_probability`");
        return unsafe { secret_env_2_transition_probability(s, a, s_p, r) };
    }

    fn reset_random_state(&mut self, _seed: u64) {
        unsafe {
            let secret_env_2_from_random_state: libloading::Symbol<
                unsafe extern "C" fn() -> *mut c_void,
            > = LIB
                .get(b"secret_env_2_from_random_state")
                .expect("Failed to load function `secret_env_2_from_random_state`");
            let env = secret_env_2_from_random_state();
            self.env = env
        }
    }

    fn available_actions_ids(&self) -> Array1<usize> {
        unsafe {
            let mut aa = Vec::new();
            let secret_env_2_available_actions: libloading::Symbol<
                unsafe extern "C" fn(*const c_void) -> *const usize,
            > = LIB
                .get(b"secret_env_2_available_actions")
                .expect("Failed to load function `secret_env_2_available_actions`");
            let actions = secret_env_2_available_actions(self.env);

            let secret_env_2_available_actions_len: libloading::Symbol<
                unsafe extern "C" fn(*const c_void) -> usize,
            > = LIB
                .get(b"secret_env_2_available_actions_len")
                .expect("Failed to load function `secret_env_2_available_actions_len`");

            // show all available actions
            for i in 0..secret_env_2_available_actions_len(self.env) {
                aa.push(*actions.add(i));
            }
            Array1::from_vec(aa)
        }
    }

    /*
    fn is_forbidden(&self, action: usize) -> bool {
        let secret_env_2_is_forbidden: libloading::Symbol<
            unsafe extern "C" fn(*const c_void, usize) -> bool,
        > = unsafe { LIB.get(b"secret_env_2_is_forbidden") }
            .expect("Failed to load function `secret_env_2_is_forbidden`");
        return unsafe { secret_env_2_is_forbidden(self.env, action) };
    }
     */

    fn available_action_delete(&self) {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        let secret_env_2_is_game_over: libloading::Symbol<
            unsafe extern "C" fn(*const c_void) -> bool,
        > = unsafe { LIB.get(b"secret_env_2_is_game_over") }
            .expect("Failed to load function `secret_env_2_is_game_over`");
        return unsafe { secret_env_2_is_game_over(self.env) };
    }

    fn step(&mut self, action: usize) {
        unsafe {
            let secret_env_2_step: libloading::Symbol<unsafe extern "C" fn(*mut c_void, usize)> =
                LIB.get(b"secret_env_2_step")
                    .expect("Failed to load function `secret_env_2_step`");
            secret_env_2_step(self.env, action);
        }
    }

    fn delete(&mut self) {
        unsafe {
            let secret_env_2_delete: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = LIB
                .get(b"secret_env_2_delete")
                .expect("Failed to load function `secret_env_2_delete`");
            secret_env_2_delete(self.env)
        }
    }

    fn score(&self) -> f32 {
        unsafe {
            let secret_env_2_score: libloading::Symbol<unsafe extern "C" fn(*const c_void) -> f32> =
                LIB.get(b"secret_env_2_score")
                    .expect("Failed to load function `secret_env_2_score`");
            secret_env_2_score(self.env)
        }
    }

    fn display(&self) {
        unsafe {
            let secret_env_2_display: libloading::Symbol<unsafe extern "C" fn(*const c_void)> = LIB
                .get(b"secret_env_2_display")
                .expect("Failed to load function `secret_env_2_display`");
            secret_env_2_display(self.env)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let secret0 = SecretEnv2::default();
        dbg!(secret0.env);
        dbg!(SecretEnv2::num_states());
        assert_eq!(SecretEnv2::num_states(), 8192);
        dbg!(SecretEnv2::num_actions());
        assert_eq!(SecretEnv2::num_actions(), 3);
        dbg!(SecretEnv2::num_rewards());
        assert_eq!(SecretEnv2::num_rewards(), 3);
        for i in 0..SecretEnv2::num_rewards() {
            dbg!(SecretEnv2::get_reward(i));
        }
        assert_eq!(SecretEnv2::get_reward(0), -1.0);
        assert_eq!(SecretEnv2::get_reward(1), 0.0);
        assert_eq!(SecretEnv2::get_reward(2), 1.0);

        let mut env = SecretEnv2::default();
        dbg!(env.state_id());

        assert_eq!(
            SecretEnv2::build_transition_probability( 0, 0, 0, 0),
            0.0
        );

        let secret_env_2_state_id: libloading::Symbol<
            unsafe extern "C" fn(*const c_void) -> usize,
        > = unsafe { LIB.get(b"secret_env_2_state_id") }
            .expect("Failed to load function `secret_env_2_state_id`");
        unsafe {
            dbg!(secret_env_2_state_id(env.env));
        }

        unsafe {
            let secret_env_2_new: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> = LIB
                .get(b"secret_env_2_new")
                .expect("Failed to load function `secret_env_2_new`");
            let env2_p = secret_env_2_new();

            let env3 = SecretEnv2::default();
            dbg!(env3.env);
            dbg!(env3.state_id());
            dbg!(secret_env_2_state_id(env3.env));

            let env3_p = env3.env;

            dbg!(env2_p);
            dbg!(env3_p);
            println!("{:?}", env3_p);
            println!("{:?}", env2_p);

            dbg!(secret_env_2_state_id(env2_p));
        }

        assert_eq!(env.state_id(), 0);
        assert_eq!(env.is_terminal(), false);
    }
}
