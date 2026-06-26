mod model;

use model::encoder;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn main() {
    let mut e = encoder::Encoder::new(8, 4, 64, 8, 1, &mut SmallRng::seed_from_u64(0u64));
    print!("{:?}", e.forward(vec![0, 1]));
}
