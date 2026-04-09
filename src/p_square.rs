use anyhow::{Result, anyhow};

#[derive(Debug)]
struct MarkerPosition {
    position: f64,
    increment: f64,
}

impl MarkerPosition {
    pub fn new(position: f64, increment: f64) -> Self {
        MarkerPosition { position, increment }
    }

    pub fn increment(&mut self) {
        self.position += self.increment;
    }
}

fn between(x: &f64, low: &f64, high: &f64, include_high: bool) -> bool {
    if include_high {
        return x >= low && x <= high;
    }
    return x >= low && x < high;
}

fn parabolic_height(index: usize, d_sign: f64, marker_positions: &Vec<MarkerPosition>, marker_heights: &Vec<f64>) -> f64 {
    let a = d_sign / (marker_positions[index + 1].position - marker_positions[index - 1].position);
    
    let b = marker_positions[index].position - marker_positions[index - 1].position + d_sign;
    let c = (marker_heights[index + 1] - marker_heights[index]) / (marker_positions[index + 1].position - marker_positions[index].position);

    let d = marker_positions[index + 1].position - marker_positions[index].position - d_sign;
    let e = (marker_heights[index] - marker_heights[index - 1]) / (marker_positions[index].position - marker_positions[index - 1].position);

    return marker_heights[index] + a * (b * c + d * e);
}

fn linear_height(index: usize, d_sign: f64, marker_positions: &Vec<MarkerPosition>, marker_heights: &Vec<f64>) -> f64 {
    let d_sign_index = (index as f64 + d_sign) as usize;
    let ratio = (marker_heights[d_sign_index] - marker_heights[index]) / (marker_positions[d_sign_index].position - marker_positions[index].position);
    return marker_heights[index] + d_sign * ratio;
}

fn adjust_middle_markers(marker_positions: &mut Vec<MarkerPosition>, desired_marker_positions: &Vec<MarkerPosition>, marker_heights: &mut Vec<f64>) {
    for index in 1..4 {
        let d = desired_marker_positions[index].position - marker_positions[index].position;
        let marker_diff_next = marker_positions[index + 1].position - marker_positions[index].position;
        let marker_diff_prev = marker_positions[index - 1].position - marker_positions[index].position;
        if (d >= 1.0 && marker_diff_next > 1.0) || (d <= -1.0 && marker_diff_prev < -1.0) {
            let d_sign = d.signum();
            let new_marker_height = parabolic_height(index, d_sign, marker_positions, marker_heights);

            if marker_heights[index - 1] < new_marker_height && new_marker_height < marker_heights[index + 1] {
                marker_heights[index] = new_marker_height;
            } else {
                marker_heights[index] = linear_height(index, d_sign, marker_positions, marker_heights);
            }

            marker_positions[index].position += d_sign;
        }
    }
}

/// Estimates a percentile value from a sequence of numbers.
/// 
/// O(n) runtime complexity, O(1) space complexity.
/// 
/// Jain, Raj, and Imrich Chlamtac.
/// "The P2 algorithm for dynamic calculation of quantiles and histograms without storing observations."
/// Communications of the ACM 28.10 (1985): 1076-1085.
/// 
/// ### Arguments
///
/// * `xs` - An iterator of f64 values to estimate the percentile of.
/// * `percentile` - The percentile to calculate, given as a float.
///                  I.e. for the median, 0.5 would be used.
///
/// ### Returns
///
/// The estimated percentile value.
pub fn p_square<'a, I>(xs: &mut I, percentile: f64) -> Result<f64> where I: Iterator<Item = &'a f64> {
    if percentile > 1.0 || percentile < 0.0 {
        return Err(anyhow!("Invalid percentile provided."));
    }

    let prefix: Vec<f64> = xs.take(5).cloned().collect();
    if prefix.len() < 5 {
        return Err(anyhow!("Not enough data was provided."));
    }

    // Construct the main data containers.
    let mut marker_heights = prefix.to_vec();
    marker_heights.sort_by(f64::total_cmp);
    let mut marker_positions: Vec<MarkerPosition> = (1..6).map(|x| MarkerPosition::new(x as f64, 1.0)).collect();
    let mut desired_marker_positions = vec![
        MarkerPosition::new(1.0, 0.0),
        MarkerPosition::new(1.0 + 2.0 * percentile, percentile / 2.0),
        MarkerPosition::new(1.0 + 4.0 * percentile, percentile),
        MarkerPosition::new(3.0 + 2.0 * percentile, (percentile + 1.0) / 2.0),
        MarkerPosition::new(5.0, 1.0),
    ];

    for x in xs {
        let k;
        if x < &marker_heights[0] {
            marker_heights[0] = *x;
            k = 1;
        } else if between(x, &marker_heights[0], &marker_heights[1], false) {
            k = 1;
        } else if between(x, &marker_heights[1], &marker_heights[2], false) {
            k = 2;
        } else if between(x, &marker_heights[2], &marker_heights[3], false) {
            k = 3;
        } else if between(x, &marker_heights[3], &marker_heights[4], true) {
            k = 4;
        } else {
            marker_heights[4] = *x;
            k = 4;
        }
        
        for marker_position in marker_positions.iter_mut().skip(k) {
            marker_position.increment();
        }
        for marker_position in desired_marker_positions.iter_mut() {
            marker_position.increment();
        }
        adjust_middle_markers(&mut marker_positions, &desired_marker_positions, &mut marker_heights);
    }

    return Ok(marker_heights[2]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incorrect_percentile() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = p_square(&mut xs.iter(), -0.5);
        assert!(out.is_err());

        let out = p_square(&mut xs.iter(), 1.5);
        assert!(out.is_err());

        let out = p_square(&mut xs.iter(), 1.0);
        assert!(out.is_ok());

        let out = p_square(&mut xs.iter(), 0.0);
        assert!(out.is_ok());
    }

    #[test]
    fn test_less_than_5() {
        let mut xs = Vec::new();
        let out = p_square(&mut xs.iter(), 0.5);
        assert!(out.is_err());

        for i in 0..4 {
            xs.push(i as f64);
            let out = p_square(&mut xs.iter(), 0.5);
            assert!(out.is_err());
        }

        xs.push(4.0);
        let out = p_square(&mut xs.iter(), 0.5);
        assert_eq!(out.unwrap(), 2.0);
    }

    #[test]
    fn test_full_psquare() {
        let xs = vec![
            0.02, 0.5, 0.74, 3.39, 0.83, 22.37, 10.15, 15.43, 38.62, 15.92, 34.60, 10.28, 1.47, 0.40, 0.05, 11.39, 0.27, 0.42, 0.09, 11.37
        ];
        let out = p_square(&mut xs.iter(), 0.5);
        assert_eq!(out.unwrap(), 4.246239408803644);

        let out = p_square(&mut xs.iter(), 0.75);
        assert_eq!(out.unwrap(), 17.765707208697677);

        let out = p_square(&mut xs.iter(), 0.25);
        assert_eq!(out.unwrap(), 0.3871122398589066);
    }
}