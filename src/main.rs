
mod environement;
mod reinforcement_learning_functions;
mod utils;

use std::collections::{btree_map::Entry, BTreeMap};
use termtree::{GlyphPalette, Tree};

#[derive(Debug)]
enum Node<'s> {
    Path(BTreeMap<&'s str, Node<'s>>),
    Status(&'s str),
}

const OUTPUT: &str ="

running 42 tests
test environement::line_world::tests::test_available_action ... ok
test environement::line_world::tests::test_from_random_state ... ok
test environement::line_world::tests::test_line_world ... ok
test environement::line_world::tests::test_new ... ok
test environement::monty_hall_1::tests::test_available_action ... ok
test environement::monty_hall_1::tests::test_monty_hall_1 ... ok
test environement::monty_hall_1::tests::test_new ... ok
test environement::two_round_rps::tests::test_available_action ... ok
test environement::secret_env_0::tests::test_init ... ok
test environement::two_round_rps::tests::test_new ... ok
test environement::two_round_rps::tests::test_two_round_strategy ... ok
test environement::two_round_rps::tests::test_two_round_world ... ok
test environement::two_round_rps::tests::test_from_random_state ... ok
test environement::grid_world::tests::test_line_world ... ok
test environement::grid_world::tests::test_from_random_state ... ok
test environement::grid_world::tests::test_grid_world_strategy ... ok
test environement::grid_world::tests::test_available_action ... ok
test environement::grid_world::tests::test_new ... ok
test reinforcement_learning_functions::policy_evaluation::tests::policy_evaluation_line_world ... ok
test reinforcement_learning_functions::monte_carlo_with_exploring_start::tests::monte_carlo_with_exploring_start_returns_correct_policy ... ok
test reinforcement_learning_functions::monte_carlo_off_policy::tests::monte_carlo_off_policy_lineworld ... ok
test reinforcement_learning_functions::policy_iteration::tests::policy_iteration_line_world ... ok
test reinforcement_learning_functions::policy_iteration::tests::policy_iteration_monty_hall_1 ... ok
test reinforcement_learning_functions::monte_carlo_with_exploring_start::tests::monte_carlo_with_exploring_start_monty_hall_1 ... ok
test reinforcement_learning_functions::q_learning::tests::q_learning_grid_world ... ok
test reinforcement_learning_functions::q_learning::tests::q_learning_policy_lineworld ... ok
test reinforcement_learning_functions::monte_carlo_off_policy::tests::monte_carlo_off_policy_monty_hall_1 ... ok
test reinforcement_learning_functions::q_learning::tests::q_learning_policy_two_round_rps ... ok
test reinforcement_learning_functions::sarsa::tests::sarsa_monty_hall_1 ... ok
test reinforcement_learning_functions::monte_carlo_off_policy::tests::monte_carlo_off_policy_rps ... ok
test reinforcement_learning_functions::sarsa::tests::sarsa_policy_two_round_rps ... ok
test reinforcement_learning_functions::sarsa::tests::sarsa_policy_lineworld ... ok
test reinforcement_learning_functions::monte_carlo_with_exploring_start::tests::monte_carlo_with_exploring_start_two_round_rps ... ok
test reinforcement_learning_functions::monte_carlo_off_policy::tests::monte_carlo_off_policy_grid_world ... ok
test reinforcement_learning_functions::monte_carlo_on_policy::tests::monte_carlo_on_policy_monty_hall_1 ... ok
test reinforcement_learning_functions::monte_carlo_on_policy::tests::monte_carlo_on_policy_lineworld ... ok
test reinforcement_learning_functions::monte_carlo_with_exploring_start::tests::monte_carlo_with_exploring_start_returns_correct_policy_grid_world ... ok
test reinforcement_learning_functions::monte_carlo_on_policy::tests::monte_carlo_on_policy_grid_world ... ok
test reinforcement_learning_functions::policy_iteration::tests::policy_iteration_two_round_pfs ... ok
test reinforcement_learning_functions::policy_iteration::tests::policy_iteration_env_0 has been running for over 60 seconds
test reinforcement_learning_functions::policy_iteration::tests::policy_iteration_grid_world has been running for over 60 seconds
test reinforcement_learning_functions::sarsa::tests::sarsa_policy_gridworld has been running for over 60 seconds

";

fn main() {
    println!("{}", pretty_test(OUTPUT).unwrap());
}

fn pretty_test(output: &str) -> Option<Tree<&str>> {
    let mut path = BTreeMap::new();
    for line in output.trim().lines() {
        let mut iter = line.trim().splitn(3, ' ');
        let mut split = iter.nth(1)?.split("::");
        let next = split.next();
        let status = iter.next()?;
        make_mods(split, status, &mut path, next);
    }
    let mut tree = Tree::new("test");
    for (root, child) in path {
        make_tree(root, &child, &mut tree);
    }
    Some(tree)
}

// Add paths to Node
fn make_mods<'s>(
    mut split: impl Iterator<Item = &'s str>,
    status: &'s str,
    path: &mut BTreeMap<&'s str, Node<'s>>,
    key: Option<&'s str>,
) {
    let Some(key) = key else { return };
    let next = split.next();
    match path.entry(key) {
        Entry::Vacant(empty) => {
            if next.is_some() {
                let mut btree = BTreeMap::new();
                make_mods(split, status, &mut btree, next);
                empty.insert(Node::Path(btree));
            } else {
                empty.insert(Node::Status(status));
            }
        }
        Entry::Occupied(mut node) => {
            if let Node::Path(btree) = node.get_mut() {
                make_mods(split, status, btree, next)
            }
        }
    }
}

// Add Node to Tree
fn make_tree<'s>(root: &'s str, node: &Node<'s>, parent: &mut Tree<&'s str>) {
    match node {
        Node::Path(btree) => {
            let mut t = Tree::new(root);
            for (path, child) in btree {
                make_tree(path, child, &mut t);
            }
            parent.push(t);
        }
        Node::Status(s) => {
            parent.push(Tree::new(root).with_glyphs(set_status(s)));
        }
    }
}

// Display with a status icon
fn set_status(status: &str) -> GlyphPalette {
    let mut glyph = GlyphPalette::new();
    glyph.item_indent = if status.ends_with("ok") {
        "─ ✅ "
    } else {
        "─ ❌ "
    };
    glyph
}
use std::ffi::c_void;
