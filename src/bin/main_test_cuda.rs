extern crate IABD4_reinforcement_learning;

use burn::backend::Autodiff;
use burn::backend::libtorch::LibTorchDevice;
use burn::tensor::Tensor;

type MyBackend = burn::backend::libtorch::LibTorchDevice;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {

    let device = &LibTorchDevice::Cuda(1);
    println!("Utilisation du dispositif : {:?}", device);


}