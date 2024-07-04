use std::collections::HashMap;
use std::ffi::c_void;

use ndarray::Array1;

use crate::environement::environment::Environment;
use crate::utils::lib_utils::LIB;

pub struct SecretEnv0 {
    pub env: *mut c_void,
}

impl Environment for SecretEnv0 {
    fn new() -> Self {
        let secret_env_0_new: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            unsafe { LIB.get(b"secret_env_0_new") }.expect("Failed to load function `secret_env_0_new`");
        unsafe {
            let env = secret_env_0_new();
            return SecretEnv0 { env }
        }
    }
    fn state_id(&self) -> usize {
        dbg!(self.env);
        let secret_env_0_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> = unsafe { LIB.get(b"secret_env_0_state_id") }.expect("Failed to load function `secret_env_0_state_id`");
        return unsafe { secret_env_0_state_id(self.env) };
    }

    fn num_states() -> usize {
        let secret_env_0_num_states: libloading::Symbol<unsafe extern fn() -> usize> = unsafe { LIB.get(b"secret_env_0_num_states") }.expect("Failed to load function `secret_env_0_num_states`");
        return unsafe { secret_env_0_num_states() };
    }

    fn num_actions() -> usize {
        let secret_env_0_num_actions: libloading::Symbol<unsafe extern fn() -> usize> = unsafe { LIB.get(b"secret_env_0_num_actions") }.expect("Failed to load function `secret_env_0_num_actions`");
        return unsafe { secret_env_0_num_actions() };
    }

    fn num_rewards() -> usize {
        let secret_env_0_num_rewards: libloading::Symbol<unsafe extern fn() -> usize> = unsafe { LIB.get(b"secret_env_0_num_rewards") }.expect("Failed to load function `secret_env_0_num_rewards`");
        return unsafe { secret_env_0_num_rewards() };
    }

    fn get_reward(i: usize) -> f32 {
        let secret_env_0_reward: libloading::Symbol<unsafe extern fn(usize) -> f32> = unsafe { LIB.get(b"secret_env_0_reward") }.expect("Failed to load function `secret_env_0_reward`");
        return unsafe { secret_env_0_reward(i) }
    }

    fn get_transition_probability(&mut self, s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        let secret_env_0_transition_probability: libloading::Symbol<unsafe extern fn(usize, usize, usize, usize) -> f32> = unsafe { LIB.get(b"secret_env_0_transition_probability") }.expect("Failed to load function `secret_env_0_transition_probability`");
        return unsafe { secret_env_0_transition_probability(s, a, s_p, r) }
    }

    fn reset_random_state(&mut self, seed: u64) {
        todo!("{}", seed)
    }


    fn from_random_state() -> Self {
        todo!()
    }



    fn available_action(&self) -> Array1<usize> {
        todo!()
    }

    fn available_action_delete(&self) {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        let secret_env_0_is_game_over: libloading::Symbol<unsafe extern fn(*const c_void) -> bool> = unsafe { LIB.get(b"secret_env_0_is_game_over") }.expect("Failed to load function `secret_env_0_is_game_over`");
        return unsafe { secret_env_0_is_game_over(self.env) };
    }


    fn is_forbidden(&self, state: usize) -> bool {
        let secret_env_0_is_forbidden: libloading::Symbol<unsafe extern fn(*const c_void, usize) -> bool> = unsafe { LIB.get(b"secret_env_0_is_forbidden") }.expect("Failed to load function `secret_env_0_is_forbidden`");
        return unsafe { secret_env_0_is_forbidden(self.env, state) };
    }

    fn step(&mut self, action: usize) {
        todo!("{}", action)
    }

    fn delete(&mut self) {
        todo!()
    }

    fn score(&self) -> f64 {
        todo!()
    }

    fn display(&self) {
        todo!()
    }

    fn reset(&mut self) {
        let secret_env_0_reset: libloading::Symbol<unsafe extern fn(*mut c_void)> = unsafe { LIB.get(b"secret_env_0_reset") }.expect("Failed to load function `secret_env_0_reset`");
        unsafe { secret_env_0_reset(self.env) };
    }

    fn available_actions(i: usize) -> Array1<usize> {
        todo!()
    }

    fn is_terminal_state(state: usize) -> bool {
        todo!()
    }

    fn play_strategy(&mut self, strategy: HashMap<usize, usize>) {
        todo!()
    }

    fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        let secret_env_0_transition_probability: libloading::Symbol<unsafe extern fn(usize, usize, usize, usize) -> f32> = unsafe { LIB.get(b"secret_env_0_transition_probability") }.expect("Failed to load function `secret_env_0_transition_probability`");
        return unsafe { secret_env_0_transition_probability(s, a, s_p, r) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let secret0 = SecretEnv0::new();
        dbg!(secret0.env);
        dbg!(SecretEnv0::num_states());
        assert_eq!(SecretEnv0::num_states(), 8192);
        dbg!(SecretEnv0::num_actions());
        assert_eq!(SecretEnv0::num_actions(), 3);
        dbg!(SecretEnv0::num_rewards());
        assert_eq!(SecretEnv0::num_rewards(), 3);
        for i in 0..SecretEnv0::num_rewards() {
            dbg!(SecretEnv0::get_reward(i));
        }
        assert_eq!(SecretEnv0::get_reward(0), -1.0);
        assert_eq!(SecretEnv0::get_reward(1), 0.0);
        assert_eq!(SecretEnv0::get_reward(2), 1.0);

        let mut env = SecretEnv0::new();
        dbg!(env.state_id());

        assert_eq!(SecretEnv0::get_transition_probability(&mut env,0, 0, 0, 0), 0.0);

        let secret_env_0_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
            unsafe { LIB.get(b"secret_env_0_state_id") }.expect("Failed to load function `secret_env_0_state_id`");
        unsafe {dbg!(secret_env_0_state_id(env.env));}

        unsafe {

            let secret_env_0_new: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
                LIB.get(b"secret_env_0_new").expect("Failed to load function `secret_env_0_new`");
            let env2_p = secret_env_0_new();


            let env3 = SecretEnv0::new();
            dbg!(env3.env);
            dbg!(env3.state_id());
            dbg!(secret_env_0_state_id(env3.env));

            let env3_p = env3.env;

            dbg!(env2_p);
            dbg!(env3_p);
            println!("{:?}", env3_p);
            println!("{:?}", env2_p);

            dbg!(secret_env_0_state_id(env2_p));
        }

        assert_eq!(env.state_id(), 0);
        assert_eq!(env.is_forbidden(0), true);
        assert_eq!(env.is_terminal(), false);
    }
}