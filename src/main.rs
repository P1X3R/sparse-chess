mod model;

use model::coder::{Coder, CsdrSize};
use model::encoder;
use rand::{RngExt, rngs::SmallRng};

fn main() {
    let mut rng: SmallRng = rand::make_rng();
    let e = encoder::Encoder::new(
        CsdrSize::new(4, 4, 4),
        CsdrSize::new(8, 8, 8),
        1,
        1u8,
        &mut rng,
    );
    print!(
        "{:?}",
        e.forward((0..16).map(|_| rng.random_range(0..4)).collect())
    );
}
