use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::FieldImpl;
use icicle_core::vec_ops::{VecOpsConfig, VecOps};
use icicle_runtime::memory::{DeviceVec, HostSlice};
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
    let small_poly = create_test_polynomial(small_x, small_y, 1);
    
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



// 벤치마크를 위한 크기 구조체
#[derive(Copy, Clone, Debug)]
struct PolynomialSize {
    x_size_1: usize,
    y_size_1: usize,
    x_size_2: usize,
    y_size_2: usize,
}

impl std::fmt::Display for PolynomialSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{} * {}x{}", 
               self.x_size_1, self.y_size_1, 
               self.x_size_2, self.y_size_2)
    }
}

// 벤치마크 함수
fn bench_polynomial_mul(c: &mut Criterion) {
    let sizes = vec![
        PolynomialSize { x_size_1: 8, y_size_1: 8, x_size_2: 8, y_size_2: 8 },
        PolynomialSize { x_size_1: 16, y_size_1: 16, x_size_2: 16, y_size_2: 16 },
        PolynomialSize { x_size_1: 32, y_size_1: 32, x_size_2: 32, y_size_2: 32 },
        PolynomialSize { x_size_1: 64, y_size_1: 64, x_size_2: 64, y_size_2: 64 },
    ];

    let mut group = c.benchmark_group("Polynomial_Multiplication");
    
    for size in sizes {
        // 테스트 데이터 준비
        let size_1 = size.x_size_1 * size.y_size_1;
        let size_2 = size.x_size_2 * size.y_size_2;
        
        let mut coeffs_1 = vec![ScalarField::zero(); size_1];
        let mut coeffs_2 = vec![ScalarField::zero(); size_2];
        
        for i in 0..size_1 {
            coeffs_1[i] = ScalarCfg::generate_random(1)[0];
        }
        
        for i in 0..size_2 {
            coeffs_2[i] = ScalarCfg::generate_random(1)[0];
        }
        
        let coeffs_slice_1 = HostSlice::from_slice(&coeffs_1);
        let coeffs_slice_2 = HostSlice::from_slice(&coeffs_2);
        
        // 원본 다항식 생성
        let poly_1 = DensePolynomialExt::from_coeffs(coeffs_slice_1, size.x_size_1, size.y_size_1);
        let poly_2 = DensePolynomialExt::from_coeffs(coeffs_slice_2, size.x_size_2, size.y_size_2);
        
        // Execute_program 기반 다항식 생성
        let poly_1_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice_1, size.x_size_1, size.y_size_1);
        let poly_2_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice_2, size.x_size_2, size.y_size_2);
        
        // 원본 구현 벤치마크
        group.bench_with_input(
            BenchmarkId::new("Original", size), 
            &size,
            |b, _| {
                b.iter(|| {
                    let binding = poly_1._mul(&poly_2);
                    let result = black_box(&binding);
                    // 결과 사용하여 최적화 방지
                    black_box(result.x_size + result.y_size)
                })
            }
        );
        
        // Execute_program 구현 벤치마크
        group.bench_with_input(
            BenchmarkId::new("ExecuteProgram", size), 
            &size,
            |b, _| {
                b.iter(|| {
                    let binding = poly_1_ep._mul(&poly_2_ep);
                    let result = black_box(&binding);
                    // 결과 사용하여 최적화 방지
                    black_box(result.x_size + result.y_size)
                })
            }
        );
    }
    
    group.finish();
}

fn bench_div_by_vanishing(c: &mut Criterion) {
  // 바니싱 다항식 차수 설정 - 모든 경우에 2의 거듭제곱
  let vanishing_degrees = vec![
      (4, 4, "equal_small"),
      (8, 8, "equal_medium"),
      (2, 8, "different_small"),
      (4, 16, "different_medium")
  ];
  
  // 다항식 크기 설정
  let poly_sizes = vec![
      (8, 8, "tiny"),
      (16, 16, "small"),
      (32, 32, "medium"),
      (64, 64, "large")
  ];
  
  for (poly_x, poly_y, poly_desc) in poly_sizes {
      for (denom_x, denom_y, denom_desc) in &vanishing_degrees {
          // 다항식 차수는 vanishing 다항식 차수보다 커야 함
          if poly_x <= *denom_x || poly_y <= *denom_y {
              continue;
          }
          
          // m>=2 && n==2 조건 확인 (div_by_vanishing이 지원하는 조건)
          let m = poly_x / *denom_x;
          let n = poly_y / *denom_y;
          if !(m >= 2 && n == 2) {
              continue;
          }
          
          // 벤치마크 그룹 이름 생성
          let group_name = format!("div_by_vanishing_{}_{}", poly_desc, denom_desc);
          let mut group = c.benchmark_group(&group_name);
          
          // 테스트 다항식 생성 (비영 계수 일정 수 포함)
          let num_nonzero = std::cmp::min(poly_x * poly_y / 4, 100);
          let test_poly_coeffs = create_test_coefficients(poly_x, poly_y, num_nonzero);
          let coeffs_slice = HostSlice::from_slice(&test_poly_coeffs);
          
          // 원본 구현 벤치마크
          group.bench_function("original", |b| {
              b.iter_batched(
                  // 각 반복마다 새로운 다항식 생성
                  || DensePolynomialExt::from_coeffs(coeffs_slice, poly_x, poly_y),
                  |poly| {
                      black_box(poly.div_by_vanishing(
                          black_box(*denom_x as i64), 
                          black_box(*denom_y as i64)
                      ))
                  },
                  BatchSize::SmallInput
              )
          });
          
          // Execute_program 구현 벤치마크
          group.bench_function("execute_program", |b| {
              b.iter_batched(
                  // 각 반복마다 새로운 다항식 생성
                  || DensePolynomialExtEP::from_coeffs(coeffs_slice, poly_x, poly_y),
                  |poly| {
                      black_box(poly.div_by_vanishing(
                          black_box(*denom_x as i64), 
                          black_box(*denom_y as i64)
                      ))
                  },
                  BatchSize::SmallInput
              )
          });
          
          group.finish();
      }
  }
  
  // 특별히 m>=2 && n==2 조건에 맞는 케이스들 추가
  let special_cases = vec![
      (8, 8, 4, 4),    // m=2, n=2
      (8, 16, 4, 4),   // m=2, n=4
      (8, 32, 4, 4),   // m=2, n=8
      (8, 64, 4, 4)    // m=2, n=16
  ];
  
  for (poly_x, poly_y, denom_x, denom_y) in special_cases {
      let m = poly_x / denom_x;
      let n = poly_y / denom_y;
      
      // 조건 확인
      if !(m >= 2 && n == 2) {
          continue;
      }
      
      let group_name = format!("div_by_vanishing_special_m{}_n{}", m, n);
      let mut group = c.benchmark_group(&group_name);
      
      // 테스트 다항식 생성
      let num_nonzero = std::cmp::min(poly_x * poly_y / 4, 100);
      let test_poly_coeffs = create_test_coefficients(poly_x, poly_y, num_nonzero);
      let coeffs_slice = HostSlice::from_slice(&test_poly_coeffs);
      
      // 원본 구현 벤치마크
      group.bench_function("original", |b| {
          b.iter_batched(
              || DensePolynomialExt::from_coeffs(coeffs_slice, poly_x, poly_y),
              |poly| {
                  black_box(poly.div_by_vanishing(
                      black_box(denom_x as i64), 
                      black_box(denom_y as i64)
                  ))
              },
              BatchSize::SmallInput
          )
      });
      
      // Execute_program 구현 벤치마크
      group.bench_function("execute_program", |b| {
          b.iter_batched(
              || DensePolynomialExtEP::from_coeffs(coeffs_slice, poly_x, poly_y),
              |poly| {
                  black_box(poly.div_by_vanishing(
                      black_box(denom_x as i64), 
                      black_box(denom_y as i64)
                  ))
              },
              BatchSize::SmallInput
          )
      });
      
      group.finish();
  }
  
  // 추가 테스트: div_by_vanishing과 divide_x/divide_y 비교
  let test_cases = vec![
      (16, 8, 4, 4, "small"),
      (32, 8, 4, 4, "medium"),
  ];
  
  for (poly_x, poly_y, denom_x, denom_y, desc) in test_cases {
      let group_name = format!("vanishing_vs_divide_{}", desc);
      let mut group = c.benchmark_group(&group_name);
      
      // 테스트 다항식 생성
      let num_nonzero = std::cmp::min(poly_x * poly_y / 4, 100);
      let test_poly_coeffs = create_test_coefficients(poly_x, poly_y, num_nonzero);
      let coeffs_slice = HostSlice::from_slice(&test_poly_coeffs);
      
      // div_by_vanishing 벤치마크
      group.bench_function("div_by_vanishing", |b| {
          b.iter_batched(
              || DensePolynomialExt::from_coeffs(coeffs_slice, poly_x, poly_y),
              |poly| {
                  black_box(poly.div_by_vanishing(
                      black_box(denom_x as i64), 
                      black_box(denom_y as i64)
                  ))
              },
              BatchSize::SmallInput
          )
      });
      
      group.finish();
  }
}

// 테스트용 계수 생성 함수
fn create_test_coefficients(x_size: usize, y_size: usize, num_nonzero: usize) -> Vec<ScalarField> {
  let mut coeffs = vec![ScalarField::zero(); x_size * y_size];
  
  // 비영 계수 추가
  for i in 0..num_nonzero {
      let x = (i * 3) % x_size;
      let y = (i * 7) % y_size;
      coeffs[x + y * x_size] = ScalarCfg::generate_random(1)[0];
  }
  
  coeffs
}

fn create_test_polynomial2(x_size: usize, y_size: usize, num_nonzero: usize) -> (DensePolynomial, DensePolynomialExt, DensePolynomialExtEP) {
  let mut coeffs = vec![ScalarField::zero(); x_size * y_size];
  
  // 영이 아닌 계수 추가
  for i in 0..num_nonzero {
      let x = (i * 3) % x_size;
      let y = (i * 7) % y_size;
      coeffs[x + y * x_size] = ScalarField::one();
  }
  
  let dense_poly = DensePolynomial::from_coeffs(HostSlice::from_slice(&coeffs), x_size * y_size);
  let biv_poly = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), x_size, y_size);
  let biv_poly_ep = DensePolynomialExtEP::from_coeffs(HostSlice::from_slice(&coeffs), x_size, y_size);
  
  (dense_poly, biv_poly, biv_poly_ep)
}
fn bench_to_rou_evals(c: &mut Criterion) {
  // 코셋 값 설정
  let coset_x_val = ScalarField::from_u32(7u32);
  let coset_y_val = ScalarField::from_u32(11u32);
  
  // 작은 크기 벤치마크
  let small_x = 8;
  let small_y = 8;
  let (_, small_poly, small_poly_ep) = create_test_polynomial2(small_x, small_y, 5);
  
  let mut group = c.benchmark_group("to_rou_evals_small");
  
  // 코셋 없음
  group.bench_function("original_no_coset", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(small_x * small_y).unwrap(),
          |mut evals| {
              small_poly.to_rou_evals(None, None, &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.bench_function("execute_program_no_coset", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(small_x * small_y).unwrap(),
          |mut evals| {
              small_poly_ep.to_rou_evals(None, None, &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  // X 코셋만
  group.bench_function("original_x_coset", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(small_x * small_y).unwrap(),
          |mut evals| {
              small_poly.to_rou_evals(Some(&coset_x_val), None, &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.bench_function("execute_program_x_coset", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(small_x * small_y).unwrap(),
          |mut evals| {
              small_poly_ep.to_rou_evals(Some(&coset_x_val), None, &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  // Y 코셋만
  group.bench_function("original_y_coset", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(small_x * small_y).unwrap(),
          |mut evals| {
              small_poly.to_rou_evals(None, Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.bench_function("execute_program_y_coset", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(small_x * small_y).unwrap(),
          |mut evals| {
              small_poly_ep.to_rou_evals(None, Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  // 두 코셋 모두
  group.bench_function("original_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(small_x * small_y).unwrap(),
          |mut evals| {
              small_poly.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.bench_function("execute_program_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(small_x * small_y).unwrap(),
          |mut evals| {
              small_poly_ep.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.finish();
  
  // 중간 크기 벤치마크
  let medium_x = 16;
  let medium_y = 16;
  let (_, medium_poly, medium_poly_ep) = create_test_polynomial2(medium_x, medium_y, 10);
  
  let mut group = c.benchmark_group("to_rou_evals_medium");
  
  // 두 코셋 모두
  group.bench_function("original_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(medium_x * medium_y).unwrap(),
          |mut evals| {
              medium_poly.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.bench_function("execute_program_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(medium_x * medium_y).unwrap(),
          |mut evals| {
              medium_poly_ep.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.finish();
  
  // 큰 크기 벤치마크
  let large_x = 32;
  let large_y = 32;
  let (_, large_poly, large_poly_ep) = create_test_polynomial2(large_x, large_y, 20);
  
  let mut group = c.benchmark_group("to_rou_evals_large");
  
  // 두 코셋 모두
  group.bench_function("original_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(large_x * large_y).unwrap(),
          |mut evals| {
              large_poly.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.bench_function("execute_program_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(large_x * large_y).unwrap(),
          |mut evals| {
              large_poly_ep.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.finish();
  
  // 불균형 크기 벤치마크
  let unbalanced_x = 8;
  let unbalanced_y = 32;
  let (_, unbalanced_poly, unbalanced_poly_ep) = create_test_polynomial2(unbalanced_x, unbalanced_y, 15);
  
  let mut group = c.benchmark_group("to_rou_evals_unbalanced");
  
  // 두 코셋 모두
  group.bench_function("original_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(unbalanced_x * unbalanced_y).unwrap(),
          |mut evals| {
              unbalanced_poly.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.bench_function("execute_program_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(unbalanced_x * unbalanced_y).unwrap(),
          |mut evals| {
              unbalanced_poly_ep.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.finish();
  
  // 더 큰 크기 벤치마크
  let very_large_x = 64;
  let very_large_y = 64;
  let (_, very_large_poly, very_large_poly_ep) = create_test_polynomial2(very_large_x, very_large_y, 30);
  
  let mut group = c.benchmark_group("to_rou_evals_very_large");
  
  // 두 코셋 모두
  group.bench_function("original_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(very_large_x * very_large_y).unwrap(),
          |mut evals| {
              very_large_poly.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.bench_function("execute_program_both_cosets", |b| {
      b.iter_batched(
          || DeviceVec::<ScalarField>::device_malloc(very_large_x * very_large_y).unwrap(),
          |mut evals| {
              very_large_poly_ep.to_rou_evals(Some(&coset_x_val), Some(&coset_y_val), &mut evals);
          },
          BatchSize::SmallInput,
      )
  });
  
  group.finish();
}

use std::time::Duration;


// 원본 구현과 execute_program 구현 비교 벤치마크
fn benchmark_compare_implementations(c: &mut Criterion) {
    let sizes = vec![(8, 8), (16, 16), (32, 32)];
    
    let mut group = c.benchmark_group("div_by_ruffini_implementation_comparison");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);
    
    for (x_size, y_size) in sizes {
        // 테스트용 다항식 생성
        let coeffs = ScalarCfg::generate_random(x_size * y_size);
        
        // 원본 구현용 다항식
        let poly_original = DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&coeffs),
            x_size,
            y_size,
        );
        
        // execute_program 구현용 다항식
        let poly_ep = DensePolynomialExtEP::from_coeffs(
            HostSlice::from_slice(&coeffs),
            x_size,
            y_size,
        );
        
        let x = ScalarCfg::generate_random(1)[0];
        let y = ScalarCfg::generate_random(1)[0];
        
        // 원본 구현 벤치마크
        group.bench_with_input(
            BenchmarkId::new(format!("original_{}x{}", x_size, y_size), ""),
            &(x_size, y_size),
            |b, _| {
                b.iter(|| {
                    black_box(poly_original.div_by_ruffini(black_box(x), black_box(y)))
                });
            },
        );
        
        // execute_program 구현 벤치마크
        group.bench_with_input(
            BenchmarkId::new(format!("execute_program_{}x{}", x_size, y_size), ""),
            &(x_size, y_size),
            |b, _| {
                b.iter(|| {
                    black_box(poly_ep.div_by_ruffini(black_box(x), black_box(y)))
                });
            },
        );
    }
    
    group.finish();
}

// 실행 시간에 따른 스케일링 분석 벤치마크
fn benchmark_scaling(c: &mut Criterion) {
    // 다양한 크기로 스케일링 분석
    let x_sizes = vec![4, 8, 16, 32, 64, 128];
    let y_sizes = vec![4, 8, 16, 32];
    
    let mut group = c.benchmark_group("div_by_ruffini_scaling");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(5);
    
    // X 크기 스케일링 (Y 고정)
    let fixed_y = 8;
    for x_size in &x_sizes {
        let coeffs = ScalarCfg::generate_random(*x_size * fixed_y);
        let poly = DensePolynomialExtEP::from_coeffs(
            HostSlice::from_slice(&coeffs),
            *x_size,
            fixed_y,
        );
        
        let x = ScalarCfg::generate_random(1)[0];
        let y = ScalarCfg::generate_random(1)[0];
        
        group.bench_with_input(
            BenchmarkId::new(format!("x_scaling_{}x{}", x_size, fixed_y), ""),
            x_size,
            |b, _| {
                b.iter(|| {
                    black_box(poly.div_by_ruffini(black_box(x), black_box(y)))
                });
            },
        );
    }
    
    // Y 크기 스케일링 (X 고정)
    let fixed_x = 8;
    for y_size in &y_sizes {
        let coeffs = ScalarCfg::generate_random(fixed_x * *y_size);
        let poly = DensePolynomialExtEP::from_coeffs(
            HostSlice::from_slice(&coeffs),
            fixed_x,
            *y_size,
        );
        
        let x = ScalarCfg::generate_random(1)[0];
        let y = ScalarCfg::generate_random(1)[0];
        
        group.bench_with_input(
            BenchmarkId::new(format!("y_scaling_{}x{}", fixed_x, y_size), ""),
            y_size,
            |b, _| {
                b.iter(|| {
                    black_box(poly.div_by_ruffini(black_box(x), black_box(y)))
                });
            },
        );
    }
    
    group.finish();
}


criterion_group!(
  benches, 
  bench_find_degree, 
//   bench_from_rou_evals,
  // bench_resize,
  // bench_polynomial_mul,
  // bench_div_by_vanishing,
//   bench_to_rou_evals,
//   benchmark_compare_implementations
);
criterion_main!(benches);