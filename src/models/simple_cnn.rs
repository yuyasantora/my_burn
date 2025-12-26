use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct SimpleCnn<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
    pool_size: usize,
}

#[derive(Config, Debug)]
pub struct SimpleCnnConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = 8)]
    pool_size: usize,
    #[config(default = 0.5)]
    dropout: f64,
}

impl SimpleCnnConfig {
    /// Create a new SimpleCnnConfig
    pub fn new(num_classes: usize, hidden_size: usize) -> Self {
        Self {
            num_classes,
            hidden_size,
            pool_size: 8,
            dropout: 0.5,
        }
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set pool size
    pub fn with_pool_size(mut self, pool_size: usize) -> Self {
        self.pool_size = pool_size;
        self
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimpleCnn<B> {
        let pool_size = self.pool_size;
        let linear1_input = 128 * pool_size * pool_size;

        SimpleCnn {
            conv1: Conv2dConfig::new([3, 32], [3, 3]).init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3]).init(device),
            conv3: Conv2dConfig::new([64, 128], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([pool_size, pool_size]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(linear1_input, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            pool_size,
        }
    }
}

impl<B: Backend> SimpleCnn<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _, _, _] = images.dims();

        let x = self.conv1.forward(images);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv2.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv3.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 128 * self.pool_size * self.pool_size]);

        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}
