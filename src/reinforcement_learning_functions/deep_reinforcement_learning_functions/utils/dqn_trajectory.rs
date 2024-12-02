pub mod trajectory {
    use burn::prelude::Tensor;
    use burn::tensor::backend::AutodiffBackend;
    use burn::tensor::Int;
    use rand::seq::index::sample;

    #[derive(Debug)]
    pub struct Trajectory<B: AutodiffBackend<FloatElem = f32, IntElem = i64>> {
        pub max_size: usize,
        pub len: usize,
        pub index: usize,
        pub s_t: Vec<Tensor<B, 1>>,
        pub a_t: Vec<f32>,
        pub r_t: Vec<f32>,
        pub s_p_t: Vec<Tensor<B, 1>>,
        pub a_p_t: Vec<f32>,
        pub is_terminal: Vec<f32>,
    }

    impl<B: AutodiffBackend<FloatElem = f32, IntElem = i64>> Trajectory<B> {
        pub(crate) fn new(max_size: usize) -> Self {
            Trajectory {
                max_size,
                len: 0,
                index: 0,
                s_t: Vec::with_capacity(max_size),
                a_t: Vec::with_capacity(max_size),
                r_t: Vec::with_capacity(max_size),
                s_p_t: Vec::with_capacity(max_size),
                a_p_t: Vec::with_capacity(max_size),
                is_terminal: Vec::with_capacity(max_size),
            }
        }

        pub fn push(
            &mut self,
            s_t: Tensor<B, 1>,
            a_t: f32,
            r_t: f32,
            s_p_t: Tensor<B, 1>,
            a_p_t: f32,
            is_term: f32,
        ) {
           

            self.index = (self.index + 1) % self.max_size;
            if self.len < self.max_size {
                self.len += 1;
                self.s_t.push(s_t);
                self.a_t.push(a_t);
                self.r_t.push(r_t);
                self.s_p_t.push(s_p_t);
                self.a_p_t.push(a_p_t);
                self.is_terminal.push(is_term);
            }
            else {
                self.s_t[self.index] = s_t;
                self.a_t[self.index] = a_t;
                self.r_t[self.index] = r_t;
                self.s_p_t[self.index] = s_p_t;
                self.a_p_t[self.index] =  a_p_t;
                self.is_terminal[self.index] = is_term;
            }
        }

        pub fn get_batch(&mut self, batch_size: usize) -> Self {
            //let indexes : Vec<usize> = [0..self.size].choose_multiple(&mut rand::thread_rng(), batch_size).collect();
            let indexes: Vec<usize> =
                sample(&mut rand::thread_rng(), self.len, batch_size).into_vec();


            let mut batch = Trajectory::new(batch_size);

            for i in indexes {
                batch.push(
                    self.s_t[i].clone(),
                    self.a_t[i],
                    self.r_t[i],
                    self.s_p_t[i].clone(),
                    self.a_p_t[i],
                    self.is_terminal[i],
                );
            }
            batch
        }

        pub fn get_as_tensor(
            &mut self,
            device: &B::Device,
        ) -> (
            Tensor<B, 1, Int>,
            Tensor<B, 2>,
            Tensor<B, 1, Int>,
            Tensor<B, 2>,
            Tensor<B, 1>,
            Tensor<B, 1>,
        ) {
            (
                Tensor::from_ints(self.a_t.as_slice(), device),
                Tensor::from(Tensor::stack(self.s_t.clone(), 0)).detach(),
                Tensor::from_ints(self.a_p_t.as_slice(), device),
                Tensor::from(Tensor::stack(self.s_p_t.clone(), 0)).detach(),
                Tensor::from_floats(self.r_t.as_slice(), device).detach(),
                Tensor::from_floats(self.r_t.as_slice(), device).detach(),
            )
        }
        pub fn get_as_vector(
            &mut self,
        ) -> Vec<(Tensor<B, 1>, f32, f32, Tensor<B, 1>, f32, f32)>
         {
             let mut vec  = Vec::new();
             for i in 0..self.len {
                 vec.push((self.s_t[i].clone(), self.a_t[i], self.r_t[i], self.s_p_t[i].clone(), self.a_p_t[i], self.is_terminal[i]));
             }
             vec
        }
    }
}
/*
impl<B: AutodiffBackend<FloatElem=f32, IntElem=i64>> Iterator for Trajectory<B>{
    type Item = (Tensor::<B, 1>, usize, f32, Tensor::<B, 1>, Tensor::<B, 1>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
           None
        }
        else {
            let result = Some((&self.s_t[self.index], &self.a_t[self.index], &self.r_t[self.index], &self.log_prob_t[self.index], &self.v_t[self.index]));

            result
        }
    }
}

 */
pub mod prioritized_trajectory {
    use burn::prelude::{Tensor, TensorData};
    use burn::tensor::backend::AutodiffBackend;
    use burn::tensor::Int;
    use rand::Rng;

    const EPSILON: f32 = 1e-6;

    #[derive(Debug)]
    struct SumTree {
        tree: Vec<f32>,
        capacity: usize,
    }

    impl SumTree {
        fn new(capacity: usize) -> Self {
            let tree_size = 2 * capacity - 1;
            Self {
                tree: vec![0.0; tree_size],
                capacity,
            }
        }

        fn total(&self) -> f32 {
            self.tree[0]
        }

        fn update(&mut self, idx: usize, priority: f32) {
            let tree_idx = self.capacity - 1 + idx;
            let change = priority - self.tree[tree_idx];

            let mut idx = tree_idx;
            self.tree[idx] = priority;

            while idx > 0 {
                idx = (idx - 1) / 2;
                self.tree[idx] += change;
            }
        }

        fn get(&self, mut value: f32) -> (usize, f32) {
            let mut idx = 0;

            while idx < self.capacity - 1 {
                let left = 2 * idx + 1;
                let right = left + 1;

                if left >= self.tree.len() {
                    break;
                }

                idx = if value <= self.tree[left] {
                    left
                } else {
                    value -= self.tree[left];
                    right
                };
            }

            let data_idx = idx - (self.capacity - 1);
            // Make sure we don't return an index larger than our actual data
            let data_idx = data_idx.min(self.capacity - 1);
            (data_idx, self.tree[idx])
        }
    }

    #[derive(Debug)]
    pub struct PrioritizedTrajectory<B: AutodiffBackend<FloatElem = f32, IntElem = i64>> {
        pub max_size: usize,
        pub len: usize,
        pub index: usize,
        pub s_t: Vec<Tensor<B, 1>>,
        pub a_t: Vec<f32>,
        pub r_t: Vec<f32>,
        pub s_p_t: Vec<Tensor<B, 1>>,
        pub a_p_t: Vec<f32>,
        pub is_terminal: Vec<f32>,
        sum_tree: SumTree,
        alpha: f32,
        pub(crate) beta: f32,
        max_priority: f32,
    }

    impl<B: AutodiffBackend<FloatElem = f32, IntElem = i64>> PrioritizedTrajectory<B> {
        pub fn new(max_size: usize, alpha: f32, beta: f32) -> Self {
            PrioritizedTrajectory {
                max_size,
                len: 0,
                index: 0,
                s_t: Vec::with_capacity(max_size),
                a_t: Vec::with_capacity(max_size),
                r_t: Vec::with_capacity(max_size),
                s_p_t: Vec::with_capacity(max_size),
                a_p_t: Vec::with_capacity(max_size),
                is_terminal: Vec::with_capacity(max_size),
                sum_tree: SumTree::new(max_size),
                alpha,
                beta,
                max_priority: 1.0,
            }
        }

        pub fn push(
            &mut self,
            s_t: Tensor<B, 1>,
            a_t: f32,
            r_t: f32,
            s_p_t: Tensor<B, 1>,
            a_p_t: f32,
            is_term: f32,
        ) {
            self.index = (self.index + 1) % self.max_size;

            if self.len < self.max_size {
                self.len += 1;
                self.s_t.push(s_t);
                self.a_t.push(a_t);
                self.r_t.push(r_t);
                self.s_p_t.push(s_p_t);
                self.a_p_t.push(a_p_t);
                self.is_terminal.push(is_term);
            } else {
                self.s_t[self.index] = s_t;
                self.a_t[self.index] = a_t;
                self.r_t[self.index] = r_t;
                self.s_p_t[self.index] = s_p_t;
                self.a_p_t[self.index] = a_p_t;
                self.is_terminal[self.index] = is_term;
            }

            // New experiences get maximum priority
            self.sum_tree.update(self.index, self.max_priority.powf(self.alpha));
        }

        pub fn get_prioritized_batch(&mut self, batch_size: usize) -> (PrioritizedBatch<B>, Vec<f32>, Vec<usize>) {
            if self.len < batch_size {
                panic!("Not enough samples in memory for a batch of size {}", batch_size);
            }
            let mut batch = PrioritizedBatch::new(batch_size);
            let mut weights = Vec::with_capacity(batch_size);
            let mut indices = Vec::with_capacity(batch_size);

            let segment = self.sum_tree.total() / batch_size as f32;
            let min_prob = (self.sum_tree.total() / self.len as f32).min(EPSILON);

            for i in 0..batch_size {
                let mut rng = rand::thread_rng();
                let s = segment * i as f32 + rng.gen::<f32>() * segment;

                let (idx, priority) = self.sum_tree.get(s);
                // Ensure idx is within bounds
                let idx = idx.min(self.len - 1);
                indices.push(idx);

                let prob = priority / self.sum_tree.total();
                let weight = ((prob / min_prob).powf(-self.beta) * batch_size as f32).min(20.0);
                weights.push(weight);

                batch.push(
                    self.s_t[idx].clone(),
                    self.a_t[idx],
                    self.r_t[idx],
                    self.s_p_t[idx].clone(),
                    self.a_p_t[idx],
                    self.is_terminal[idx],
                );
            }

            (batch, weights, indices)
        }

        pub fn update_priorities(&mut self, indices: Vec<usize>, td_errors: TensorData) {
            let td_errors_vec: Vec<f32> = td_errors.to_vec().unwrap();  // Handle the Result
            for (i, &idx) in indices.iter().enumerate() {  // Use &idx to get usize value
                let error = td_errors_vec[i];
                let priority = (error.abs() + EPSILON).powf(self.alpha);
                self.max_priority = self.max_priority.max(priority);
                self.sum_tree.update(idx, priority);  // idx is now usize
            }
        }
    }

    #[derive(Debug)]
    pub struct PrioritizedBatch<B: AutodiffBackend<FloatElem = f32, IntElem = i64>> {
        pub size: usize,
        pub s_t: Vec<Tensor<B, 1>>,
        pub a_t: Vec<f32>,
        pub r_t: Vec<f32>,
        pub s_p_t: Vec<Tensor<B, 1>>,
        pub a_p_t: Vec<f32>,
        pub is_terminal: Vec<f32>,
    }

    impl<B: AutodiffBackend<FloatElem = f32, IntElem = i64>> PrioritizedBatch<B> {
        pub fn new(size: usize) -> Self {
            PrioritizedBatch {
                size,
                s_t: Vec::with_capacity(size),
                a_t: Vec::with_capacity(size),
                r_t: Vec::with_capacity(size),
                s_p_t: Vec::with_capacity(size),
                a_p_t: Vec::with_capacity(size),
                is_terminal: Vec::with_capacity(size),
            }
        }

        pub fn push(
            &mut self,
            s_t: Tensor<B, 1>,
            a_t: f32,
            r_t: f32,
            s_p_t: Tensor<B, 1>,
            a_p_t: f32,
            is_term: f32,
        ) {
            self.s_t.push(s_t);
            self.a_t.push(a_t);
            self.r_t.push(r_t);
            self.s_p_t.push(s_p_t);
            self.a_p_t.push(a_p_t);
            self.is_terminal.push(is_term);
        }

        pub fn get_as_tensor(
            &mut self,
            device: &B::Device,
        ) -> (
            Tensor<B, 1, Int>,
            Tensor<B, 2>,
            Tensor<B, 1, Int>,
            Tensor<B, 2>,
            Tensor<B, 1>,
            Tensor<B, 1>,
        ) {
            (
                Tensor::from_ints(self.a_t.as_slice(), device),
                Tensor::from(Tensor::stack(self.s_t.clone(), 0)).detach(),
                Tensor::from_ints(self.a_p_t.as_slice(), device),
                Tensor::from(Tensor::stack(self.s_p_t.clone(), 0)).detach(),
                Tensor::from_floats(self.r_t.as_slice(), device).detach(),
                Tensor::from_floats(self.is_terminal.as_slice(), device).detach(),
            )
        }
    }
}