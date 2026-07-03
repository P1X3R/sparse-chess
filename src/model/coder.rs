#[derive(Debug, Copy, Clone)]
pub struct CsdrSize {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub cols: usize,
    pub flat: usize,
}

impl CsdrSize {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        return Self {
            x,
            y,
            z,
            cols: x * y,
            flat: x * y * z,
        };
    }
}

#[macro_export]
macro_rules! flat_index {
    ([$d0:expr $(, $d_tail:expr)*], [$i0:expr $(, $i_tail:expr)*]) => {
        $crate::flat_index!(@internal ($i0), [$($d_tail),*], [$($i_tail),*])
    };

    (@internal ($acc:expr), [$d_head:expr $(, $d_tail:expr)*], [$i_head:expr $(, $i_tail:expr)*]) => {
        $crate::flat_index!(@internal (($acc) * ($d_head) + ($i_head)), [$($d_tail),*], [$($i_tail),*])
    };

    (@internal ($acc:expr), [], []) => {
        $acc
    };

    (@internal ($acc:expr), $tt1:tt, $tt2:tt) => {
        compile_error!("Mismatched number of dimensions and indices in flat_index!")
    };
}
