// benches/bench_scaled_outer_product.rs

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, black_box};
use icicle_bls12_381::curve::{ScalarField, ScalarCfg, G1Affine, CurveCfg};
use icicle_core::{curve::Curve, traits::{FieldImpl, GenerateRandom}};
use libs::iotools::{           
    scaled_outer_product_1d,  
    scaled_outer_product_1d_ep,
    G1serde,
};

fn bench_scaled_outer(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaled_outer_product_1d");
    let sizes = [128, 256, 512, 1024];

    for &size in sizes.iter() {
        let (columns, rows): (Box<[ScalarField]>, Box<[ScalarField]>) =
            (ScalarCfg::generate_random(size).into_boxed_slice(),
             ScalarCfg::generate_random(size).into_boxed_slice());

        let result_template: Box<[G1serde]> = vec![G1serde::zero(); size * size].into_boxed_slice();
        let generator = CurveCfg::generate_random_affine_points(1)[0];

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &(columns.clone(), rows.clone(), generator, result_template.clone()),
            |b, (columns, rows, generator, result)| {
                b.iter(|| {
                    let mut result = result.clone();
                    scaled_outer_product_1d(
                        black_box(columns),
                        black_box(rows),
                        black_box(&generator),
                        black_box(None),
                        black_box(&mut result),
                    );
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("icicle_execute", size),
            &(columns, rows, generator, result_template),
            |b, (columns, rows, generator, result)| {
                b.iter(|| {
                    let mut result = result.clone();
                    scaled_outer_product_1d_ep(
                        black_box(columns),
                        black_box(rows),
                        black_box(&generator),
                        black_box(None),
                        black_box(&mut result),
                    );
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_scaled_outer);
criterion_main!(benches);