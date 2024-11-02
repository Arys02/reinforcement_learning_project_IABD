pub mod trajectory {
    use burn::prelude::Tensor;
    use burn::tensor::backend::AutodiffBackend;
    use rand::seq::index::sample;

    #[derive(Debug)]
    pub struct Trajectory<B: AutodiffBackend<FloatElem = f32, IntElem = i64>> {
        pub max_size: usize,
        pub size: usize,
        pub s_t: Vec<Tensor<B, 1>>,
        pub a_t: Vec<usize>,
        pub r_t: Vec<f32>,
        pub log_prob_t: Vec<Tensor<B, 1>>,
        pub v_t: Vec<Tensor<B, 1>>,
        //index: usize,
    }

    impl<B: AutodiffBackend<FloatElem = f32, IntElem = i64>> Trajectory<B> {
        pub(crate) fn new(max_size: usize) -> Self {
            Trajectory {
                max_size,
                size: 0,
                s_t: Vec::with_capacity(max_size),
                a_t: Vec::with_capacity(max_size),
                r_t: Vec::with_capacity(max_size),
                log_prob_t: Vec::with_capacity(max_size),
                v_t: Vec::with_capacity(max_size),
                //index : 0,
            }
        }

        pub fn push(
            &mut self,
            s_t: Tensor<B, 1>,
            a_t: usize,
            r_t: f32,
            log_prob_t: Tensor<B, 1>,
            v_t: Tensor<B, 1>,
        ) {
            if self.size + 1 > self.max_size {
                panic!("error size to long")
            }
            self.size += 1;
            self.s_t.push(s_t);
            self.a_t.push(a_t);
            self.r_t.push(r_t);
            self.log_prob_t.push(log_prob_t);
            self.v_t.push(v_t);
        }

        pub fn get_batch(&mut self, batch_size: usize) -> Self {
            if batch_size > self.size {
                panic!("batch size greater than size");
            }
            //let indexes : Vec<usize> = [0..self.size].choose_multiple(&mut rand::thread_rng(), batch_size).collect();
            let indexes: Vec<usize> =
                sample(&mut rand::thread_rng(), self.size, batch_size).into_vec();

            let mut batch = Trajectory::new(batch_size);

            for i in indexes {
                batch.push(
                    self.s_t[i].clone(),
                    self.a_t[i],
                    self.r_t[i],
                    self.log_prob_t[i].clone(),
                    self.v_t[i].clone(),
                );
            }
            batch
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

    #[cfg(test)]
    mod tests {
        use super::*;
        use burn::backend::Autodiff;

        type MyBackend = burn_tch::LibTorch;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        // Alias pour simplifier les types dans les tests
        type B = MyAutodiffBackend;

        #[test]
        fn test_new_trajectory() {
            let max_size = 10;
            let trajectory = Trajectory::<B>::new(max_size);

            assert_eq!(trajectory.max_size, max_size);
            assert_eq!(trajectory.size, 0);
            assert_eq!(trajectory.s_t.capacity(), max_size);
            assert_eq!(trajectory.a_t.capacity(), max_size);
            assert_eq!(trajectory.r_t.capacity(), max_size);
            assert_eq!(trajectory.log_prob_t.capacity(), max_size);
            assert_eq!(trajectory.v_t.capacity(), max_size);
        }

        #[test]
        fn test_push_trajectory() {
            let max_size = 5;

            let device = &Default::default();
            let mut trajectory = Trajectory::<B>::new(max_size);

            for i in 0..max_size {
                let s_t = Tensor::<B, 1>::from_floats([0.], device);
                let a_t = i;
                let r_t = i as f32 * 0.1;
                let log_prob_t = Tensor::<B, 1>::from_floats([0.5], device);
                let v_t = Tensor::<B, 1>::from_floats([1.0], device);

                trajectory.push(s_t, a_t, r_t, log_prob_t, v_t);

                assert_eq!(trajectory.size, i + 1);
                assert_eq!(trajectory.s_t.len(), i + 1);
                assert_eq!(trajectory.a_t.len(), i + 1);
                assert_eq!(trajectory.r_t.len(), i + 1);
                assert_eq!(trajectory.log_prob_t.len(), i + 1);
                assert_eq!(trajectory.v_t.len(), i + 1);
            }
        }

        #[test]
        fn test_get_batch() {
            let max_size = 10;
            let batch_size = 5;

            let device = &Default::default();
            let mut trajectory = Trajectory::<B>::new(max_size);

            // Remplir la trajectoire
            for i in 0..max_size {
                let s_t = Tensor::<B, 1>::from_floats([0.], device);
                let a_t = i;
                let r_t = i as f32 * 0.1;
                let log_prob_t = Tensor::<B, 1>::from_floats([0.5], device);
                let v_t = Tensor::<B, 1>::from_floats([1.0], device);

                trajectory.push(s_t, a_t, r_t, log_prob_t, v_t);
            }

            // Obtenir un batch
            let batch = trajectory.get_batch(batch_size);
            println!("{:?}", batch);

            assert_eq!(batch.size, batch_size);
            assert_eq!(batch.s_t.len(), batch_size);
            assert_eq!(batch.a_t.len(), batch_size);
            assert_eq!(batch.r_t.len(), batch_size);
            assert_eq!(batch.log_prob_t.len(), batch_size);
            assert_eq!(batch.v_t.len(), batch_size);
        }
    }
}
