# mini-running-statistics

Computes mean, variance, and standard deviation of a stream of numbers in a single pass using Welford's algorithm.

## Usage

```rust
use mini_running_statistics::RunningStats;

let mut stats = RunningStats::new();
stats.push(2.0);
stats.push(4.0);
stats.push(4.0);
stats.push(4.0);
stats.push(5.0);
stats.push(5.0);
stats.push(7.0);
stats.push(9.0);

assert_eq!(stats.count(), 8);
assert!((stats.mean().unwrap() - 5.0).abs() < 1e-12);
assert!((stats.variance().unwrap() - 4.571428571428571).abs() < 1e-12);
assert!((stats.std_dev().unwrap() - 2.138089935299395).abs() < 1e-12);
```

The library exposes the following public API:

- `RunningStats::new()` – create an empty accumulator.
- `push(x: f64)` – add a value.
- `count() -> u64` – number of values seen.
- `mean() -> Option<f64>` – arithmetic mean, `None` if empty.
- `variance() -> Option<f64>` – sample variance (Bessel's correction), `None` if fewer than two values.
- `std_dev() -> Option<f64>` – sample standard deviation, `None` if fewer than two values.
- `population_variance() -> Option<f64>` – population variance, `None` if empty.
- `population_std_dev() -> Option<f64>` – population standard deviation, `None` if empty.
- `clear()` – reset to empty state.

## Why this library exists

The naive way to compute variance is to accumulate the sum of squares and subtract the square of the sum. That formula is mathematically correct but numerically disastrous for large values with small variation: the subtraction cancels most of the significant digits. Welford's algorithm avoids that by maintaining a running mean and a running sum of squared deviations from that mean. The trade-off is a slightly more involved update step and a tiny bit more state (three numbers instead of two), but the result is stable across a wide range of inputs.

## Awkward edge

Sample variance is undefined for a single value, so `variance()` and `std_dev()` return `None` when the count is less than two. If you want the population variance (divide by `n`), use `population_variance()`; it returns `Some(0.0)` for a single value. Be explicit about which one you need.
