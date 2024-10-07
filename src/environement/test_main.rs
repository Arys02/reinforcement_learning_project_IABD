use IABD4_reinforcement_learning::environement::farkle::farkle::Farkle;
use IABD4_reinforcement_learning::environement::environment_traits::Playable;

fn main() {
    let mut farkle = Farkle::default();
    farkle.board = [0, 2, 0, 0, 1, 0];
    farkle.play_with_human();

    // Use Debug formatting
    //println!("{:?}", farkle);
}