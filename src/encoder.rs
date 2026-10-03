use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::{
    coder::{CsdrSize, FieldEntry, LocalField, ReceptiveField},
    flat_index,
};

type LearningField = ReceptiveField<u32>;

const BYTE_INV: f32 = 1.0 / 255.0;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EncoderVisibleParams {
    pub visible_size: CsdrSize,
    pub radius: i16,
}

#[derive(Serialize, Deserialize)]
pub struct EncoderVisibleSnapshot<'a> {
    visible_size: CsdrSize,
    radius: i16,

    #[serde(borrow)]
    weights: Cow<'a, [u8]>,
}

#[derive(Serialize, Deserialize)]
pub struct EncoderSnapshot<'a> {
    hidden_size: CsdrSize,
    visible_layers: Vec<EncoderVisibleSnapshot<'a>>,

    learning_radius: isize,

    #[serde(borrow)]
    hidden_totals: Cow<'a, [u16]>,

    #[serde(borrow)]
    is_committed: Cow<'a, [bool]>,

    choice: f32,
    vigilance: f32,
    active_ratio: f32,
    lr: f32,
}

#[derive(Debug)]
pub(crate) struct EncoderVisibleLayer {
    pub(crate) visible_size: CsdrSize,
    radius: i16,
    area: usize,

    receptive_field: LocalField,
    weights: Box<[u8]>,
}

impl EncoderVisibleLayer {
    fn new(visible_size: CsdrSize, hidden_size: &CsdrSize, radius: i16) -> Self {
        let diameter = radius * 2 + 1;
        let area = (diameter * diameter) as usize;

        let receptive_field = LocalField::new(
            hidden_size,
            &visible_size,
            radius,
            Self::local_field_weight_idx(&visible_size, hidden_size, area),
        );

        let mut rng = fastrand::Rng::new();

        let weight_count = visible_size.z * hidden_size.cols * area * hidden_size.z;

        let weights = std::iter::repeat_with(|| rng.u8(0..=8))
            .take(weight_count)
            .collect();

        Self {
            visible_size,
            radius,
            area,
            receptive_field,
            weights,
        }
    }

    #[inline]
    fn local_field_weight_idx(
        _visible_size: &CsdrSize,
        hidden_size: &CsdrSize,
        area: usize,
    ) -> impl Fn(usize, usize) -> u32 {
        move |hidden_col: usize, in_field_idx: usize| {
            flat_index!(
                [_visible_size.z, hidden_size.cols, area, hidden_size.z],
                [0, hidden_col, in_field_idx, 0]
            ) as u32
        }
    }

    #[inline(always)]
    fn get_weight_idx(
        &self,
        hidden_size: &CsdrSize,
        visible_z: usize,
        hidden_col: usize,
        area_idx: usize,
        hidden_z: usize,
    ) -> usize {
        flat_index!(
            [
                self.visible_size.z,
                hidden_size.cols,
                self.area,
                hidden_size.z
            ],
            [visible_z, hidden_col, area_idx, hidden_z]
        )
    }

    #[inline(always)]
    fn calc_hidden_sum(
        &self,
        hidden_size: &CsdrSize,
        input: &[u16],
        local_field: &[FieldEntry],
        sum_col: &mut [u32],
    ) {
        debug_assert_eq!(input.len(), self.visible_size.cols);
        debug_assert!(
            input
                .iter()
                .all(|&cell| (cell as usize) < self.visible_size.z)
        );

        for field in local_field {
            let input_cell = input[field.input_cell_idx as usize] as usize;

            let weights_start =
                field.weights_base as usize + self.get_weight_idx(hidden_size, input_cell, 0, 0, 0);

            let weights_end = weights_start + hidden_size.z;

            let weights_col = &self.weights[weights_start..weights_end];

            for cell in 0..hidden_size.z {
                sum_col[cell] += weights_col[cell] as u32;
            }
        }
    }

    #[inline]
    fn learn(
        &mut self,
        hidden_size: &CsdrSize,
        input: &[u16],
        hidden_col: usize,
        hidden_z: usize,
        is_committed: bool,
        weight_deltas: &[u8; 256],
        hidden_total: &mut u16,
    ) {
        let local_field = self.receptive_field.get_col(hidden_col);

        for field in local_field {
            let input_cell = input[field.input_cell_idx as usize] as usize;

            let weights_idx = field.weights_base as usize
                + self.get_weight_idx(hidden_size, input_cell, 0, 0, hidden_z);

            let old = self.weights[weights_idx];

            let new = if is_committed {
                weight_deltas[old as usize]
            } else {
                255 - old
            };

            self.weights[weights_idx] = new;

            *hidden_total += (new - old) as u16;
        }
    }

    fn snapshot(&self) -> EncoderVisibleSnapshot<'_> {
        EncoderVisibleSnapshot {
            visible_size: self.visible_size,
            radius: self.radius,
            weights: Cow::Borrowed(&self.weights),
        }
    }

    fn from_snapshot(snapshot: EncoderVisibleSnapshot, hidden_size: &CsdrSize) -> Self {
        let diameter = snapshot.radius * 2 + 1;
        let area = (diameter * diameter) as usize;

        let receptive_field = LocalField::new(
            hidden_size,
            &snapshot.visible_size,
            snapshot.radius,
            Self::local_field_weight_idx(&snapshot.visible_size, hidden_size, area),
        );

        Self {
            visible_size: snapshot.visible_size,
            radius: snapshot.radius,
            area,
            receptive_field,
            weights: snapshot.weights.into_owned().into_boxed_slice(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EncoderLearningData {
    max_activations: Box<[f32]>,
    learn_flags: Box<[bool]>,
}

#[derive(Debug)]
pub struct Encoder {
    pub(crate) hidden_size: CsdrSize,
    pub(crate) visible_layers: Box<[EncoderVisibleLayer]>,

    learning_radius: isize,
    learning_field: LearningField,

    weight_deltas: [u8; 256],
    counts_all: Box<[u16]>,
    counts_except: Box<[u16]>,

    hidden_totals: Box<[u16]>,
    is_committed: Box<[bool]>,

    choice: f32,
    vigilance: f32,
    active_ratio: f32,
    lr: f32,
}

impl Encoder {
    pub fn new(
        hidden_size: CsdrSize,
        visible_params: &[EncoderVisibleParams],
        learning_radius: isize,
        lr: f32,
        choice: f32,
        vigilance: f32,
        active_ratio: f32,
    ) -> Self {
        assert!(!visible_params.is_empty());

        let visible_layers = visible_params
            .iter()
            .map(|params| {
                EncoderVisibleLayer::new(params.visible_size, &hidden_size, params.radius)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let counts_all = (0..hidden_size.cols)
            .map(|col| Self::calc_count_all(col, &visible_layers) as u16)
            .collect();
        let counts_except = (0..hidden_size.cols)
            .map(|col| Self::calc_count_except(col, &visible_layers) as u16)
            .collect();
        let learning_field = Self::init_learning_field_lut(&hidden_size, learning_radius);

        Self {
            hidden_size,
            visible_layers,

            learning_radius,
            learning_field,

            weight_deltas: Self::compute_deltas(lr),
            counts_all,
            counts_except,

            hidden_totals: vec![0; hidden_size.flat].into_boxed_slice(),

            is_committed: vec![false; hidden_size.flat].into_boxed_slice(),

            choice,
            vigilance,
            active_ratio,
            lr,
        }
    }

    fn calc_count_all(hidden_col: usize, visible_layers: &[EncoderVisibleLayer]) -> usize {
        visible_layers
            .iter()
            .map(|l| {
                let field_len = l.receptive_field.get_col(hidden_col).len();
                field_len * l.visible_size.z
            })
            .sum::<usize>()
    }

    fn calc_count_except(hidden_col: usize, visible_layers: &[EncoderVisibleLayer]) -> usize {
        visible_layers
            .iter()
            .map(|l| {
                let field_len = l.receptive_field.get_col(hidden_col).len();
                field_len * (l.visible_size.z - 1)
            })
            .sum::<usize>()
    }

    #[inline]
    fn compute_deltas(lr: f32) -> [u8; 256] {
        std::array::from_fn(|w| (w as u8).saturating_add((lr * (255.0 - w as f32)).ceil() as u8))
    }

    fn init_learning_field_lut(hidden_size: &CsdrSize, learning_radius: isize) -> LearningField {
        let diameter = (2 * learning_radius + 1) as usize;
        let area = diameter * diameter;

        let mut learning_field_lut = Vec::with_capacity(hidden_size.cols * area);

        let mut learning_field_offsets = Vec::with_capacity(hidden_size.cols);

        for hidden_col in 0..hidden_size.cols {
            let hidden_x = hidden_col % hidden_size.x;
            let hidden_y = hidden_col / hidden_size.x;

            let start_idx = learning_field_lut.len() as u32;

            for delta_y in -learning_radius..=learning_radius {
                let neighbor_y = hidden_y as isize + delta_y;

                if neighbor_y < 0 || neighbor_y >= hidden_size.y as isize {
                    continue;
                }

                for delta_x in -learning_radius..=learning_radius {
                    let neighbor_x = hidden_x as isize + delta_x;

                    if neighbor_x < 0
                        || neighbor_x >= hidden_size.x as isize
                        || (delta_x == 0 && delta_y == 0)
                    {
                        continue;
                    }

                    let neighbor_idx = flat_index!(
                        [hidden_size.y, hidden_size.x],
                        [neighbor_y as usize, neighbor_x as usize]
                    );

                    learning_field_lut.push(neighbor_idx as u32);
                }
            }

            let end_idx = learning_field_lut.len() as u32;

            learning_field_offsets.push((start_idx, end_idx));
        }

        LearningField {
            lut: learning_field_lut.into_boxed_slice(),
            offsets: learning_field_offsets.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn can_col_learn(&self, hidden_col: usize, learning_data: &EncoderLearningData) -> bool {
        if !learning_data.learn_flags[hidden_col] {
            return false;
        }

        let learning_field = self.learning_field.get_col(hidden_col);
        let center_activation = learning_data.max_activations[hidden_col];
        let field_cnt = learning_field.len();
        let allowed = (self.active_ratio * field_cnt as f32) as usize;
        let exceeded = learning_field
            .iter()
            .filter(|&&n| learning_data.max_activations[n as usize] > center_activation)
            .take(allowed + 1)
            .count()
            > allowed;
        !exceeded
    }

    pub fn forward(&self, inputs: &[&[u16]]) -> (Box<[u16]>, EncoderLearningData) {
        assert_eq!(inputs.len(), self.visible_layers.len());
        for (input, layer) in inputs.iter().zip(self.visible_layers.iter()) {
            assert_eq!(input.len(), layer.visible_size.cols);
            debug_assert!(
                input
                    .iter()
                    .all(|&cell| { (cell as usize) < layer.visible_size.z })
            );
        }

        let mut hidden_sum = vec![0; self.hidden_size.flat].into_boxed_slice();
        let mut hidden = vec![0; self.hidden_size.cols].into_boxed_slice();
        let mut cols_max_activation = vec![0.0; self.hidden_size.cols].into_boxed_slice();
        let mut cols_learn_flag = vec![false; self.hidden_size.cols].into_boxed_slice();

        for hidden_col in 0..self.hidden_size.cols {
            let sum_col_start =
                flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);
            let sum_col_end = sum_col_start + self.hidden_size.z;
            let sum_col = &mut hidden_sum[sum_col_start..sum_col_end];

            for (input, layer) in inputs.iter().zip(self.visible_layers.iter()) {
                let local_field = layer.receptive_field.get_col(hidden_col);
                layer.calc_hidden_sum(&self.hidden_size, input, local_field, sum_col);
            }
        }

        for hidden_col in 0..self.hidden_size.cols {
            let sum_col_start =
                flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);

            let sum_col_end = sum_col_start + self.hidden_size.z;
            let sum_col = &hidden_sum[sum_col_start..sum_col_end];
            let total_col = &self.hidden_totals[sum_col_start..sum_col_end];
            let committed_col = &self.is_committed[sum_col_start..sum_col_end];

            let count_all = self.counts_all[hidden_col] as f32;
            let count_except = self.counts_except[hidden_col] as f32;
            let count_except_inv = 1.0 / count_except;
            let beta = self.choice + count_all;

            let mut max_activation = 0.0;
            let mut max_complete_activation = 0.0;

            let mut max_activation_cell = None;
            let mut max_complete_activation_cell = 0;

            for cell in 0..self.hidden_size.z {
                let sum = sum_col[cell] as f32 * BYTE_INV;
                let total = total_col[cell] as f32 * BYTE_INV;
                let complemented = sum - total + count_except;
                let match_score = complemented * count_except_inv;
                let activation = complemented / (beta - total);

                if (!committed_col[cell] || match_score >= self.vigilance)
                    && activation > max_activation
                {
                    max_activation = activation;
                    max_activation_cell = Some(cell);
                }

                if activation > max_complete_activation {
                    max_complete_activation = activation;
                    max_complete_activation_cell = cell;
                }
            }

            match max_activation_cell {
                None => {
                    hidden[hidden_col] = max_complete_activation_cell as u16;
                    cols_max_activation[hidden_col] = max_complete_activation;
                    cols_learn_flag[hidden_col] = false;
                }
                Some(cell) => {
                    hidden[hidden_col] = cell as u16;
                    cols_max_activation[hidden_col] = max_activation;
                    cols_learn_flag[hidden_col] = true;
                }
            }
        }

        (
            hidden,
            EncoderLearningData {
                max_activations: cols_max_activation,
                learn_flags: cols_learn_flag,
            },
        )
    }

    pub fn learn(
        &mut self,
        inputs: &[&[u16]],
        hidden: &[u16],
        learning_data: &EncoderLearningData,
    ) {
        assert_eq!(inputs.len(), self.visible_layers.len());
        assert_eq!(hidden.len(), self.hidden_size.cols);
        for (input, layer) in inputs.iter().zip(self.visible_layers.iter()) {
            assert_eq!(input.len(), layer.visible_size.cols);
        }

        for hidden_col in 0..self.hidden_size.cols {
            if !self.can_col_learn(hidden_col, learning_data) {
                continue;
            }

            let hidden_z = hidden[hidden_col] as usize;

            let hidden_idx = flat_index!(
                [self.hidden_size.cols, self.hidden_size.z],
                [hidden_col, hidden_z]
            );

            let is_committed = self.is_committed[hidden_idx];

            for (layer, input) in self.visible_layers.iter_mut().zip(inputs.iter()) {
                layer.learn(
                    &self.hidden_size,
                    input,
                    hidden_col,
                    hidden_z,
                    is_committed,
                    &self.weight_deltas,
                    &mut self.hidden_totals[hidden_idx],
                );
            }

            self.is_committed[hidden_idx] = true;
        }
    }

    pub fn get_commited_rate(&self) -> f32 {
        self.is_committed
            .iter()
            .filter(|&&committed| committed)
            .count() as f32
            / self.is_committed.len() as f32
    }

    pub fn get_saturated_rate(&self) -> f32 {
        self.visible_layers
            .iter()
            .map(|l| {
                l.weights.iter().filter(|w| **w == 255).count() as f32 / l.weights.len() as f32
            })
            .sum::<f32>()
            / self.visible_layers.len() as f32
    }

    pub fn get_snapshot(&self) -> EncoderSnapshot<'_> {
        EncoderSnapshot {
            hidden_size: self.hidden_size,

            visible_layers: self
                .visible_layers
                .iter()
                .map(|layer| layer.snapshot())
                .collect(),

            learning_radius: self.learning_radius,

            hidden_totals: Cow::Borrowed(&self.hidden_totals),

            is_committed: Cow::Borrowed(&self.is_committed),

            choice: self.choice,
            vigilance: self.vigilance,
            active_ratio: self.active_ratio,
            lr: self.lr,
        }
    }

    pub fn from_snapshot(snapshot: EncoderSnapshot) -> Self {
        let hidden_size = snapshot.hidden_size;

        let visible_layers = snapshot
            .visible_layers
            .into_iter()
            .map(|layer| EncoderVisibleLayer::from_snapshot(layer, &hidden_size))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let counts_all = (0..hidden_size.cols)
            .map(|col| Self::calc_count_all(col, &visible_layers) as u16)
            .collect();
        let counts_except = (0..hidden_size.cols)
            .map(|col| Self::calc_count_except(col, &visible_layers) as u16)
            .collect();

        Self {
            hidden_size,

            visible_layers,

            learning_radius: snapshot.learning_radius,

            learning_field: Self::init_learning_field_lut(&hidden_size, snapshot.learning_radius),

            weight_deltas: Self::compute_deltas(snapshot.lr),
            counts_all,
            counts_except,

            hidden_totals: snapshot.hidden_totals.into_owned().into_boxed_slice(),

            is_committed: snapshot.is_committed.into_owned().into_boxed_slice(),

            choice: snapshot.choice,
            vigilance: snapshot.vigilance,
            active_ratio: snapshot.active_ratio,
            lr: snapshot.lr,
        }
    }
}
