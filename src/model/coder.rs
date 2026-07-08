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

#[derive(Debug)]
pub(crate) struct FieldBounds {
    pub(crate) field_start_x: i16,
    pub(crate) field_start_y: i16,

    pub(crate) clamped_start_x: i16,
    pub(crate) clamped_start_y: i16,

    pub(crate) clamped_end_x: i16,
    pub(crate) clamped_end_y: i16,
}

impl FieldBounds {
    #[inline]
    pub(crate) fn new(
        x: usize,
        y: usize,
        radius: i16,
        ratio: (f32, f32),
        projected_csdr_size: &CsdrSize,
    ) -> Self {
        let visible_center_x = ((x as f32 + 0.5f32) * ratio.0) as i16;
        let visible_center_y = ((y as f32 + 0.5f32) * ratio.1) as i16;

        let field_start_x = visible_center_x - radius;
        let field_start_y = visible_center_y - radius;

        let field_end_x = visible_center_x + radius;
        let field_end_y = visible_center_y + radius;

        let clamped_start_x = field_start_x.max(0);
        let clamped_start_y = field_start_y.max(0);

        let clamped_end_x = field_end_x.min(projected_csdr_size.x as i16 - 1);
        let clamped_end_y = field_end_y.min(projected_csdr_size.y as i16 - 1);

        Self {
            field_start_x,
            field_start_y,
            clamped_start_x,
            clamped_start_y,
            clamped_end_x,
            clamped_end_y,
        }
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

#[inline(always)]
pub fn rand_round(x: f32, rng: &mut fastrand::Rng) -> f32 {
    if rng.f32_inclusive() < x.fract() {
        x.floor()
    } else {
        x.ceil()
    }
}
