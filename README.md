# Project Title

This project implements various reinforcement learning algorithms and environments in Rust. It includes several modules for different environments, reinforcement learning functions, and utility functions. The project interacts with external libraries to manage specific environments.

## Project Structure

```
.
├── Cargo.lock
├── Cargo.toml
├── libs
│   ├── libsecret_envs.dylib
│   ├── libsecret_envs.so
│   ├── libsecret_envs_intel_macos.dylib
│   └── secret_envs.dll
├── record
└── src
    ├── environement
    │   ├── environment.rs
    │   ├── grid_world.rs
    │   ├── line_world.rs
    │   ├── mod.rs
    │   ├── monty_hall_1.rs
    │   ├── secret_env_0.rs
    │   ├── secret_env_1.rs
    │   ├── secret_env_2.rs
    │   ├── secret_env_3.rs
    │   └── two_round_rps.rs
    ├── lib.rs
    ├── main.rs
    ├── notebook
    │   ├── cours_drl.ipynb
    │   └── env_timer_comparaison.ipynb
    ├── reinforcement_learning_functions
    │   ├── mod.rs
    │   ├── monte_carlo_off_policy.rs
    │   ├── monte_carlo_on_policy.rs
    │   ├── monte_carlo_with_exploring_start.rs
    │   ├── policy_evaluation.rs
    │   ├── policy_iteration.rs
    │   ├── q_learning.rs
    │   └── sarsa.rs
    └── utils
        ├── csv_utils.rs
        ├── lib_utils.rs
        ├── main2.rs
        ├── mod.rs
        └── policy_utils.rs
```

## Description

### Libraries

- `libs`: Contains dynamic libraries (`.dylib`, `.so`, `.dll`) used by the project for secret environments.

### Records

- `record`: Contains CSV files with the results of various reinforcement learning algorithms executed on different environments.

### Source Code

- `src/environement`: Contains Rust implementations for different environments.
    - `environment.rs`: Definition of the `Environment` trait.
    - `grid_world.rs`, `line_world.rs`, `monty_hall_1.rs`, `secret_env_0.rs`, `secret_env_1.rs`, `secret_env_2.rs`, `secret_env_3.rs`, `two_round_rps.rs`: Specific implementations of different environments.

- `src/lib.rs`: The main library file for the project.

- `src/main.rs`: The main executable file for running the project.

- `src/notebook`: Contains Jupyter Notebooks for exploring and comparing different environments.
    - `cours_drl.ipynb`: Notebook for the DRL course.
    - `env_timer_comparaison.ipynb`: Notebook for environment timer comparison.

- `src/reinforcement_learning_functions`: Contains implementations of various reinforcement learning algorithms.
    - `monte_carlo_off_policy.rs`, `monte_carlo_on_policy.rs`, `monte_carlo_with_exploring_start.rs`, `policy_evaluation.rs`, `policy_iteration.rs`, `q_learning.rs`, `sarsa.rs`: Specific implementations of reinforcement learning algorithms.

- `src/utils`: Utility functions and helpers.
    - `csv_utils.rs`, `lib_utils.rs`, `main2.rs`, `mod.rs`, `policy_utils.rs`: Various utility functions and helpers.

## Usage

### Building the Project

To build the project, run:
```sh
cargo build
```

### Running the Project

To run the main executable, use:
```sh
cargo run
```

### Testing the Project

To run the tests, use:
```sh
cargo test
```

### Generating Documentation

To generate the project documentation, run:
```sh
cargo doc --open
```

## Notebooks

The project includes Jupyter Notebooks for additional analysis and visualization. You can run these notebooks using Jupyter:

```sh
jupyter notebook
```

Navigate to the `src/notebook` directory and open the desired notebook.