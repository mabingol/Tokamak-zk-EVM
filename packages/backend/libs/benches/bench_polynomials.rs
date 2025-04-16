use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::FieldImpl;
use icicle_runtime::memory::HostSlice;
use libs::polynomials::{BivariatePolynomial, DensePolynomialExt};
use libs::polynomials_ep::{BivariatePolynomialEP, DensePolynomialExtEP};
use icicle_core::traits::GenerateRandom;

fn create_test_polynomial(x_size: usize, y_size: usize, num_nonzero: usize) -> DensePolynomial {
    let mut coeffs = vec![ScalarField::zero(); x_size * y_size];
    
    // Add non-zero coefficients
    for i in 0..num_nonzero {
        let x = (i * 3) % x_size;
        let y = (i * 7) % y_size;
        coeffs[x + y * x_size] = ScalarField::one();
    }
    
    DensePolynomial::from_coeffs(HostSlice::from_slice(&coeffs), x_size * y_size)
}

// 랜덤 평가값을 생성하는 함수
fn create_random_evals(size: usize) -> Vec<ScalarField> {
  let evals = ScalarCfg::generate_random(size);
    
    evals
}

fn bench_find_degree(c: &mut Criterion) {
    // Small polynomial benchmark
    let small_x = 64;
    let small_y = 128;
    let small_poly = create_test_polynomial(small_x, small_y, 20);
    
    let mut group = c.benchmark_group("find_degree_small");
    group.bench_function("original", |b| {
        b.iter(|| DensePolynomialExt::find_degree(black_box(&small_poly), black_box(small_x), black_box(small_y)))
    });
    
    group.bench_function("execute_program", |b| {
        b.iter(|| DensePolynomialExtEP::find_degree(black_box(&small_poly), black_box(small_x), black_box(small_y)))
    });
    group.finish();
    
    // Reversed dimensions benchmark
    let mut group = c.benchmark_group("find_degree_reversed_dims");
    group.bench_function("original", |b| {
        b.iter(|| DensePolynomialExt::find_degree(black_box(&small_poly), black_box(small_y), black_box(small_x)))
    });
    
    group.bench_function("execute_program", |b| {
        b.iter(|| DensePolynomialExtEP::find_degree(black_box(&small_poly), black_box(small_y), black_box(small_x)))
    });
    group.finish();
    
    // Large polynomial benchmark
    let large_x = 256;
    let large_y = 256;
    let large_poly = create_test_polynomial(large_x, large_y, 50);
    
    let mut group = c.benchmark_group("find_degree_large");
    group.bench_function("original", |b| {
        b.iter(|| DensePolynomialExt::find_degree(black_box(&large_poly), black_box(large_x), black_box(large_y)))
    });
    
    group.bench_function("execute_program", |b| {
        b.iter(|| DensePolynomialExtEP::find_degree(black_box(&large_poly), black_box(large_x), black_box(large_y)))
    });
    group.finish();
    
    // Very large polynomial benchmark
    let very_large_x = 512;
    let very_large_y = 512;
    let very_large_poly = create_test_polynomial(very_large_x, very_large_y, 100);
    
    let mut group = c.benchmark_group("find_degree_very_large");
    group.bench_function("original", |b| {
        b.iter(|| DensePolynomialExt::find_degree(black_box(&very_large_poly), black_box(very_large_x), black_box(very_large_y)))
    });
    
    group.bench_function("execute_program", |b| {
        b.iter(|| DensePolynomialExtEP::find_degree(black_box(&very_large_poly), black_box(very_large_x), black_box(very_large_y)))
    });
    group.finish();
}

fn bench_from_rou_evals(c: &mut Criterion) {
    // 코셋 값 설정
    let coset_x_val = ScalarField::from_u32(7u32);
    let coset_y_val = ScalarField::from_u32(11u32);
    
    // 작은 크기 벤치마크
    let small_x = 8;
    let small_y = 8;
    let small_size = small_x * small_y;
    let small_evals = create_random_evals(small_size);
    
    let mut group = c.benchmark_group("from_rou_evals_small");
    
    // 코셋 없음
    group.bench_function("original_no_coset", |b| {
        b.iter(|| DensePolynomialExt::from_rou_evals(
            black_box(HostSlice::from_slice(&small_evals)),
            black_box(small_x),
            black_box(small_y),
            black_box(None),
            black_box(None)
        ))
    });
    
    group.bench_function("execute_program_no_coset", |b| {
        b.iter(|| DensePolynomialExtEP::from_rou_evals(
            black_box(HostSlice::from_slice(&small_evals)),
            black_box(small_x),
            black_box(small_y),
            black_box(None),
            black_box(None)
        ))
    });
    
    // X 코셋만
    group.bench_function("original_x_coset", |b| {
        b.iter(|| DensePolynomialExt::from_rou_evals(
            black_box(HostSlice::from_slice(&small_evals)),
            black_box(small_x),
            black_box(small_y),
            black_box(Some(&coset_x_val)),
            black_box(None)
        ))
    });
    
    group.bench_function("execute_program_x_coset", |b| {
        b.iter(|| DensePolynomialExtEP::from_rou_evals(
            black_box(HostSlice::from_slice(&small_evals)),
            black_box(small_x),
            black_box(small_y),
            black_box(Some(&coset_x_val)),
            black_box(None)
        ))
    });
    
    // Y 코셋만
    group.bench_function("original_y_coset", |b| {
        b.iter(|| DensePolynomialExt::from_rou_evals(
            black_box(HostSlice::from_slice(&small_evals)),
            black_box(small_x),
            black_box(small_y),
            black_box(None),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.bench_function("execute_program_y_coset", |b| {
        b.iter(|| DensePolynomialExtEP::from_rou_evals(
            black_box(HostSlice::from_slice(&small_evals)),
            black_box(small_x),
            black_box(small_y),
            black_box(None),
            black_box(Some(&coset_y_val))
        ))
    });
    
    // 두 코셋 모두
    group.bench_function("original_both_cosets", |b| {
        b.iter(|| DensePolynomialExt::from_rou_evals(
            black_box(HostSlice::from_slice(&small_evals)),
            black_box(small_x),
            black_box(small_y),
            black_box(Some(&coset_x_val)),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.bench_function("execute_program_both_cosets", |b| {
        b.iter(|| DensePolynomialExtEP::from_rou_evals(
            black_box(HostSlice::from_slice(&small_evals)),
            black_box(small_x),
            black_box(small_y),
            black_box(Some(&coset_x_val)),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.finish();
    
    // 중간 크기 벤치마크
    let medium_x = 16;
    let medium_y = 16;
    let medium_size = medium_x * medium_y;
    let medium_evals = create_random_evals(medium_size);
    
    let mut group = c.benchmark_group("from_rou_evals_medium");
    
    // 두 코셋 모두
    group.bench_function("original_both_cosets", |b| {
        b.iter(|| DensePolynomialExt::from_rou_evals(
            black_box(HostSlice::from_slice(&medium_evals)),
            black_box(medium_x),
            black_box(medium_y),
            black_box(Some(&coset_x_val)),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.bench_function("execute_program_both_cosets", |b| {
        b.iter(|| DensePolynomialExtEP::from_rou_evals(
            black_box(HostSlice::from_slice(&medium_evals)),
            black_box(medium_x),
            black_box(medium_y),
            black_box(Some(&coset_x_val)),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.finish();
    
    // 큰 크기 벤치마크
    let large_x = 32;
    let large_y = 32;
    let large_size = large_x * large_y;
    let large_evals = create_random_evals(large_size);
    
    let mut group = c.benchmark_group("from_rou_evals_large");
    
    // 두 코셋 모두
    group.bench_function("original_both_cosets", |b| {
        b.iter(|| DensePolynomialExt::from_rou_evals(
            black_box(HostSlice::from_slice(&large_evals)),
            black_box(large_x),
            black_box(large_y),
            black_box(Some(&coset_x_val)),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.bench_function("execute_program_both_cosets", |b| {
        b.iter(|| DensePolynomialExtEP::from_rou_evals(
            black_box(HostSlice::from_slice(&large_evals)),
            black_box(large_x),
            black_box(large_y),
            black_box(Some(&coset_x_val)),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.finish();
    
    // 불균형 크기 벤치마크
    let unbalanced_x = 8;
    let unbalanced_y = 32;
    let unbalanced_size = unbalanced_x * unbalanced_y;
    let unbalanced_evals = create_random_evals(unbalanced_size);
    
    let mut group = c.benchmark_group("from_rou_evals_unbalanced");
    
    // 두 코셋 모두
    group.bench_function("original_both_cosets", |b| {
        b.iter(|| DensePolynomialExt::from_rou_evals(
            black_box(HostSlice::from_slice(&unbalanced_evals)),
            black_box(unbalanced_x),
            black_box(unbalanced_y),
            black_box(Some(&coset_x_val)),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.bench_function("execute_program_both_cosets", |b| {
        b.iter(|| DensePolynomialExtEP::from_rou_evals(
            black_box(HostSlice::from_slice(&unbalanced_evals)),
            black_box(unbalanced_x),
            black_box(unbalanced_y),
            black_box(Some(&coset_x_val)),
            black_box(Some(&coset_y_val))
        ))
    });
    
    group.finish();
}

fn bench_resize(c: &mut Criterion) {
  // Define polynomial sizes to benchmark
  let test_cases = vec![
      // (original_x_size, original_y_size, target_x_size, target_y_size, description)
      (8, 8, 16, 16, "small_expand"),
      (16, 16, 8, 8, "small_shrink"),
      (8, 16, 16, 8, "small_reshape"),
      (32, 32, 64, 64, "medium_expand"),
      (64, 64, 32, 32, "medium_shrink"),
      (32, 64, 64, 32, "medium_reshape"),
      (128, 128, 256, 256, "large_expand"),
      (256, 256, 128, 128, "large_shrink"),
  ];
  
  for (orig_x, orig_y, target_x, target_y, desc) in test_cases {
      let mut group = c.benchmark_group(format!("resize_{}", desc));
      
      // Create test coefficients
      let size = orig_x * orig_y;
      let mut coeffs = vec![ScalarField::zero(); size];
      
      // Fill with random values
      for i in 0..size {
          coeffs[i] = ScalarCfg::generate_random(1)[0];
      }
      
      // Create HostSlice from coefficients
      let coeffs_slice = HostSlice::from_slice(&coeffs);
      
      // Benchmark original implementation
      group.bench_function("original", |b| {
          b.iter_batched(
              // Setup for each iteration
              || DensePolynomialExt::from_coeffs(black_box(coeffs_slice), black_box(orig_x), black_box(orig_y)),
              // Benchmark operation
              |mut poly| {
                  poly.resize(black_box(target_x), black_box(target_y));
                  poly
              },
              BatchSize::SmallInput
          )
      });
      
      // Benchmark execute_program implementation
      group.bench_function("execute_program", |b| {
          b.iter_batched(
              // Setup for each iteration
              || DensePolynomialExtEP::from_coeffs(black_box(coeffs_slice), black_box(orig_x), black_box(orig_y)),
              // Benchmark operation
              |mut poly| {
                  poly.resize(black_box(target_x), black_box(target_y));
                  poly
              },
              BatchSize::SmallInput
          )
      });
      
      group.finish();
  }
  
  // Special case: resizing with same dimensions
  let mut group = c.benchmark_group("resize_same_size");
  
  // Create test coefficients
  let orig_x = 64;
  let orig_y = 64;
  let target_x = 64;
  let target_y = 64;
  let size = orig_x * orig_y;
  let mut coeffs = vec![ScalarField::zero(); size];
  
  // Fill with random values
  for i in 0..size {
      coeffs[i] = ScalarCfg::generate_random(1)[0];
  }
  
  // Create HostSlice from coefficients
  let coeffs_slice = HostSlice::from_slice(&coeffs);
  
  // Benchmark original implementation
  group.bench_function("original", |b| {
      b.iter_batched(
          // Setup for each iteration
          || DensePolynomialExt::from_coeffs(black_box(coeffs_slice), black_box(orig_x), black_box(orig_y)),
          // Benchmark operation
          |mut poly| {
              poly.resize(black_box(target_x), black_box(target_y));
              poly
          },
          BatchSize::SmallInput
      )
  });
  
  // Benchmark execute_program implementation
  group.bench_function("execute_program", |b| {
      b.iter_batched(
          // Setup for each iteration
          || DensePolynomialExtEP::from_coeffs(black_box(coeffs_slice), black_box(orig_x), black_box(orig_y)),
          // Benchmark operation
          |mut poly| {
              poly.resize(black_box(target_x), black_box(target_y));
              poly
          },
          BatchSize::SmallInput
      )
  });
  
  group.finish();
}

criterion_group!(
  benches, 
  // bench_find_degree, 
  // bench_from_rou_evals,
  bench_resize
);
criterion_main!(benches);