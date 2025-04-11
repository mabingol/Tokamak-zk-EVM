// benches/outer_product_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};
use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_core::traits::FieldImpl;

use libs::vectors::{outer_product_two_vecs, outer_product_two_vecs_rayon, outer_product_two_vecs_ep};

fn bench_outer_products(c: &mut Criterion) {
    let col_len = 512;
    let row_len = 512;
    let total = col_len * row_len;

    let col_vec: Box<[ScalarField]> = (0..col_len)
        .map(|i| ScalarField::from_u32((i % 100) as u32 + 1))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let row_vec: Box<[ScalarField]> = (0..row_len)
        .map(|i| ScalarField::from_u32(((i + 1) % 100) as u32 + 1))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let mut res_seq: Box<[ScalarField]> = vec![ScalarField::zero(); total].into_boxed_slice();
    let mut res_par: Box<[ScalarField]> = vec![ScalarField::zero(); total].into_boxed_slice();

    c.bench_function("outer_product_two_vecs", |b| {
        b.iter(|| {
            outer_product_two_vecs(&col_vec, &row_vec, &mut res_seq);
        })
    });

    c.bench_function("outer_product_two_vecs_rayon", |b| {
        b.iter(|| {
            outer_product_two_vecs_rayon(&col_vec, &row_vec, &mut res_par);
        })
    });

    c.bench_function("outer_product_two_vecs_ep", |b| {
        b.iter(|| {
            outer_product_two_vecs_ep(&col_vec, &row_vec, &mut res_par);
        })
    });
}

criterion_group!(benches, bench_outer_products);
criterion_main!(benches);
