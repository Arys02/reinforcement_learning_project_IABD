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
                sample(&mut rand::thread_rng(), self.max_size, batch_size).into_vec();


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
