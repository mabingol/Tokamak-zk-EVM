use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_runtime::memory::{HostSlice};
use icicle_core::traits::GenerateRandom;
use std::time::Duration;

use libs::polynomials::DensePolynomialExt;
use libs::polynomials::BivariatePolynomial;

// 다항식 생성 검증 함수
fn verify_from_coeffs() -> Result<(), String> {
    println!("====== 다항식 생성 검증 시작 ======");
    
    // 작은 크기로 시작
    let x_size = 2;
    let y_size = 2;
    
    // 랜덤 다항식 생성
    let poly_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(x_size * y_size);
    println!("Created coeffs of size: {}", poly_coeffs.len());
    
    // 다항식 생성
    let poly = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&poly_coeffs), 
            x_size, 
            y_size
        )
    }) {
        Ok(p) => p,
        Err(_) => return Err(format!("Failed to create polynomial with x_size={}, y_size={}", x_size, y_size))
    };
    
    println!("Successfully created polynomial with x_size={}, y_size={}", x_size, y_size);
    
    // 1x1 다항식도 생성해봄
    let small_coeffs = vec![ScalarCfg::generate_random(1)[0]];
    let small_poly = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&small_coeffs),
            1,
            1
        )
    }) {
        Ok(p) => p,
        Err(_) => return Err("Failed to create 1x1 polynomial".to_string())
    };
    
    println!("Successfully created 1x1 polynomial");
    
    // x 단변수 다항식 검증 (x_size > 1, y_size = 1)
    let x_uni_size = 4;
    let x_uni_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(x_uni_size);
    let x_uni_poly = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&x_uni_coeffs),
            x_uni_size,
            1
        )
    }) {
        Ok(p) => p,
        Err(_) => return Err(format!("Failed to create x-univariate polynomial with x_size={}", x_uni_size))
    };
    
    println!("Successfully created x-univariate polynomial with x_size={}", x_uni_size);
    
    // y 단변수 다항식 검증 (x_size = 1, y_size > 1)
    let y_uni_size = 4;
    let y_uni_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(y_uni_size);
    let y_uni_poly = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&y_uni_coeffs),
            1,
            y_uni_size
        )
    }) {
        Ok(p) => p,
        Err(_) => return Err(format!("Failed to create y-univariate polynomial with y_size={}", y_uni_size))
    };
    
    println!("Successfully created y-univariate polynomial with y_size={}", y_uni_size);
    
    // 더 큰 크기 테스트
    let big_x_size = 8;
    let big_y_size = 8;
    let big_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(big_x_size * big_y_size);
    let big_poly = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&big_coeffs),
            big_x_size,
            big_y_size
        )
    }) {
        Ok(p) => p,
        Err(_) => return Err(format!("Failed to create large polynomial with x_size={}, y_size={}", big_x_size, big_y_size))
    };
    
    println!("Successfully created large polynomial with x_size={}, y_size={}", big_x_size, big_y_size);
    
    // 코드에서 문제가 발생했던 조합 테스트
    let x_size = 8;
    let y_size = 8;
    
    // X 단변수 다항식 생성 (절반 크기)
    let denom_x_size = x_size / 2; // 4
    let denom_x_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(denom_x_size);
    let denom_x = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&denom_x_coeffs), 
            denom_x_size, 
            1
        )
    }) {
        Ok(p) => p,
        Err(_) => return Err(format!("Failed to create x denominator with x_size={}", denom_x_size))
    };
    
    println!("Successfully created x denominator with x_size={}", denom_x_size);
    
    // Y 단변수 다항식 생성 (절반 크기)
    let denom_y_size = y_size / 2; // 4
    let denom_y_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(denom_y_size);
    let denom_y = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&denom_y_coeffs), 
            1, 
            denom_y_size
        )
    }) {
        Ok(p) => p,
        Err(_) => return Err(format!("Failed to create y denominator with y_size={}", denom_y_size))
    };
    
    println!("Successfully created y denominator with y_size={}", denom_y_size);
    
    // 테스트를 늘려서 문제가 있는 크기 조합 찾기
    for size in &[2, 4, 8, 16] {
        let coeffs: Vec<ScalarField> = ScalarCfg::generate_random(*size);
        match std::panic::catch_unwind(|| {
            DensePolynomialExt::from_coeffs(
                HostSlice::from_slice(&coeffs),
                *size,
                1
            )
        }) {
            Ok(_) => println!("Size {}x1 works fine", size),
            Err(_) => return Err(format!("Failed with size {}x1", size))
        }
        
        match std::panic::catch_unwind(|| {
            DensePolynomialExt::from_coeffs(
                HostSlice::from_slice(&coeffs),
                1,
                *size
            )
        }) {
            Ok(_) => println!("Size 1x{} works fine", size),
            Err(_) => return Err(format!("Failed with size 1x{}", size))
        }
    }
    
    println!("====== 다항식 생성 검증 완료 ======");
    Ok(())
}

pub fn bench_polynomial_division(c: &mut Criterion) {
    // 먼저 from_coeffs 메서드 검증
    match verify_from_coeffs() {
        Ok(_) => println!("다항식 생성 검증 성공!"),
        Err(e) => {
            println!("다항식 생성 검증 실패: {}", e);
            return; // 벤치마크 실행 중단
        }
    }
    
    // 8x8 크기의 다항식만 사용하여 테스트
    let x_size = 8;
    let y_size = 8;
    
    let mut group = c.benchmark_group("polynomial_division");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    
    // 랜덤 다항식 생성
    let poly_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(x_size * y_size);
    println!("Main polynomial coeffs size: {}", poly_coeffs.len());
    
    let poly = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&poly_coeffs), 
            x_size, 
            y_size
        )
    }) {
        Ok(p) => p,
        Err(e) => {
            println!("메인 다항식 생성 실패!");
            return;
        }
    };
    
    println!("메인 다항식 생성 성공!");
    
    // X 단변수 다항식 생성 (절반 크기)
    let denom_x_size = x_size / 2; // 4
    let denom_x_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(denom_x_size);
    println!("X 단변수 다항식 coeffs size: {}", denom_x_coeffs.len());
    
    let denom_x = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&denom_x_coeffs), 
            denom_x_size, 
            1
        )
    }) {
        Ok(p) => p,
        Err(e) => {
            println!("X 단변수 다항식 생성 실패!");
            return;
        }
    };
    
    println!("X 단변수 다항식 생성 성공!");
    
    // Y 단변수 다항식 생성 (절반 크기)
    let denom_y_size = y_size / 2; // 4
    let denom_y_coeffs: Vec<ScalarField> = ScalarCfg::generate_random(denom_y_size);
    println!("Y 단변수 다항식 coeffs size: {}", denom_y_coeffs.len());
    
    let denom_y = match std::panic::catch_unwind(|| {
        DensePolynomialExt::from_coeffs(
            HostSlice::from_slice(&denom_y_coeffs), 
            1, 
            denom_y_size
        )
    }) {
        Ok(p) => p,
        Err(e) => {
            println!("Y 단변수 다항식 생성 실패!");
            return;
        }
    };
    
    println!("Y 단변수 다항식 생성 성공!");
    
    // 랜덤 점 (x, y) 생성
    let point_x = ScalarCfg::generate_random(1)[0];
    let point_y = ScalarCfg::generate_random(1)[0];
    
    // 1. divide_x 벤치마크
    group.bench_function(
        "divide_x", 
        |b| b.iter(|| poly.divide_x(&denom_x))
    );
    
    // // 2. divide_y 벤치마크
    // group.bench_function(
    //     "divide_y", 
    //     |b| b.iter(|| poly.divide_y(&denom_y))
    // );
    
    // 3. div_by_ruffini 벤치마크
    group.bench_function(
        "div_by_ruffini", 
        |b| b.iter(|| poly.div_by_ruffini(point_x, point_y))
    );
    
    // 4. div_by_vanishing 벤치마크
    group.bench_function(
        "div_by_vanishing", 
        |b| b.iter(|| poly.div_by_vanishing((x_size/2) as i64, (y_size/2) as i64))
    );
    
    group.finish();
}

criterion_group!(benches, bench_polynomial_division);
criterion_main!(benches);