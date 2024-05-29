mod environement;
mod reinforcement_learning_functions;
mod utils;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use crate::environement::environement::Environement;
    use crate::environement::line_world::LineWorld;
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        let mut lw = LineWorld::new();

        lw.display();
        lw.step(0);
        lw.display();
        lw.step(1);
        lw.display();
        lw.step(1);
        lw.display();

        assert_eq!(lw.state_id(), 1);
        assert_eq!(result, 4);
    }
}
