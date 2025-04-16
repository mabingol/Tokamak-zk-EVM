use criterion::{black_box, criterion_group, criterion_main, Criterion};
use icicle_bls12_381::curve::ScalarField;
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::FieldImpl;
use icicle_runtime::memory::HostSlice;
use libs::polynomials::{BivariatePolynomial, DensePolynomialExt};
use libs::polynomials_ep::{BivariatePolynomialEP, DensePolynomialExtEP};

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

criterion_group!(benches, bench_find_degree);
criterion_main!(benches);