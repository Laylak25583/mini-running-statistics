//! One-pass running statistics using Welford's algorithm.
//!
//! This crate provides a simple accumulator that computes mean, variance,
//! and standard deviation of a stream of numbers in a single pass.
//! The algorithm is numerically stable and does not require storing the
//! entire stream, making it suitable for online processing.
//!
//! # Example
//!
//! ```
//! use mini_running_statistics::RunningStats;
//!
//! let mut stats = RunningStats::new();
//! stats.push(2.0);
//! stats.push(4.0);
//! stats.push(4.0);
//! stats.push(4.0);
//! stats.push(5.0);
//! stats.push(5.0);
//! stats.push(7.0);
//! stats.push(9.0);
//!
//! assert_eq!(stats.count(), 8);
//! assert!((stats.mean().unwrap() - 5.0).abs() < 1e-12);
//! assert!((stats.variance().unwrap() - 4.571428571428571).abs() < 1e-12);
//! assert!((stats.std_dev().unwrap() - 2.138089935299395).abs() < 1e-12);
//! ```

/// Accumulates statistics for a stream of `f64` values.
///
/// Uses Welford's online algorithm, which is numerically stable and
/// avoids the cancellation errors of the naive sum-of-squares approach.
/// The state is maintained in three fields: the count, the running mean,
/// and the sum of squared differences from the current mean (`m2`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    /// Creates a new, empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Adds a single value to the stream.
    ///
    /// The update formulas are:
    /// ```text
    /// n     = n + 1
    /// delta = x - mean
    /// mean  = mean + delta / n
    /// m2    = m2 + delta * (x - mean)
    /// ```
    /// These maintain the invariant that `m2` is the sum of squared
    /// deviations from the current mean.
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Returns the number of values pushed so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the arithmetic mean of the stream, or `None` if empty.
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    /// Returns the sample variance (with Bessel's correction), or `None` if
    /// fewer than two values have been pushed.
    ///
    /// The sample variance is `m2 / (n - 1)`. For a single value the
    /// variance is undefined, hence `None`.
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m2 / (self.count - 1) as f64)
        }
    }

    /// Returns the sample standard deviation, or `None` if fewer than two
    /// values have been pushed.
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Returns the population variance (divide by `n`), or `None` if empty.
    ///
    /// This is provided for completeness; the sample variance is usually
    /// what you want.
    pub fn population_variance(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.m2 / self.count as f64)
        }
    }

    /// Returns the population standard deviation, or `None` if empty.
    pub fn population_std_dev(&self) -> Option<f64> {
        self.population_variance().map(f64::sqrt)
    }

    /// Resets the accumulator to its initial empty state.
    pub fn clear(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn empty_stats() {
        let stats = RunningStats::new();
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.variance(), None);
        assert_eq!(stats.std_dev(), None);
        assert_eq!(stats.population_variance(), None);
        assert_eq!(stats.population_std_dev(), None);
    }

    #[test]
    fn single_value() {
        let mut stats = RunningStats::new();
        stats.push(42.0);
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), Some(42.0));
        assert_eq!(stats.variance(), None);
        assert_eq!(stats.std_dev(), None);
        assert_eq!(stats.population_variance(), Some(0.0));
        assert_eq!(stats.population_std_dev(), Some(0.0));
    }

    #[test]
    fn two_values() {
        let mut stats = RunningStats::new();
        stats.push(1.0);
        stats.push(3.0);
        assert_eq!(stats.count(), 2);
        assert!(approx_eq(stats.mean().unwrap(), 2.0));
        assert!(approx_eq(stats.variance().unwrap(), 2.0));
        assert!(approx_eq(stats.std_dev().unwrap(), 2.0_f64.sqrt()));
        assert!(approx_eq(stats.population_variance().unwrap(), 1.0));
        assert!(approx_eq(stats.population_std_dev().unwrap(), 1.0));
    }

    #[test]
    fn known_dataset() {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, sample variance = 32/7 ≈ 4.571428571428571
        let mut stats = RunningStats::new();
        for &x in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.push(x);
        }
        assert_eq!(stats.count(), 8);
        assert!(approx_eq(stats.mean().unwrap(), 5.0));
        assert!(approx_eq(stats.variance().unwrap(), 32.0 / 7.0));
        assert!(approx_eq(stats.std_dev().unwrap(), (32.0_f64 / 7.0).sqrt()));
        assert!(approx_eq(stats.population_variance().unwrap(), 4.0));
        assert!(approx_eq(stats.population_std_dev().unwrap(), 2.0));
    }

    #[test]
    fn constant_values() {
        let mut stats = RunningStats::new();
        for _ in 0..10 {
            stats.push(3.14);
        }
        assert_eq!(stats.count(), 10);
        assert!(approx_eq(stats.mean().unwrap(), 3.14));
        assert!(approx_eq(stats.variance().unwrap(), 0.0));
        assert!(approx_eq(stats.std_dev().unwrap(), 0.0));
        assert!(approx_eq(stats.population_variance().unwrap(), 0.0));
        assert!(approx_eq(stats.population_std_dev().unwrap(), 0.0));
    }

    #[test]
    fn negative_and_positive() {
        let mut stats = RunningStats::new();
        stats.push(-5.0);
        stats.push(5.0);
        assert!(approx_eq(stats.mean().unwrap(), 0.0));
        assert!(approx_eq(stats.variance().unwrap(), 50.0));
        assert!(approx_eq(stats.std_dev().unwrap(), 50.0_f64.sqrt()));
    }

    #[test]
    fn large_values_stability() {
        // Welford's algorithm should handle large offsets without catastrophic cancellation.
        let mut stats = RunningStats::new();
        let base = 1e12;
        stats.push(base + 1.0);
        stats.push(base + 2.0);
        stats.push(base + 3.0);
        assert!(approx_eq(stats.mean().unwrap(), base + 2.0));
        assert!(approx_eq(stats.variance().unwrap(), 1.0));
        assert!(approx_eq(stats.std_dev().unwrap(), 1.0));
    }

    #[test]
    fn clear_resets_state() {
        let mut stats = RunningStats::new();
        stats.push(1.0);
        stats.push(2.0);
        stats.clear();
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.variance(), None);
        assert_eq!(stats.std_dev(), None);
        assert_eq!(stats.population_variance(), None);
        assert_eq!(stats.population_std_dev(), None);
        // After clear, pushing again should work as if new.
        stats.push(10.0);
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), Some(10.0));
    }
}
