// benches/bench_scaled_outer_product.rs

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, black_box};
use icicle_bls12_381::curve::{ScalarField, ScalarCfg, G1Affine, CurveCfg};
use icicle_core::{curve::Curve, traits::{FieldImpl, GenerateRandom}};
use libs::iotools::{           // 실제 크레이트 이름으로 바꿔주세요
    scaled_outer_product_1d,  // ICICLE execute_program 을 쓰지 않는 버전
    scaled_outer_product_1d_ep, // execute_program 버전
    G1serde,
};

fn bench_scaled_outer(c: &mut Criterion) {
  let mut group = c.benchmark_group("scaled_outer_product_1d");
  for &n in &[256usize, 512, 1024] {
      // ─── 벡터 길이 ───
      let L = n;
      let M = n;
      // 결과 길이
      let size = L * M;

      // 1) 랜덤 벡터 생성
      let col: Box<[ScalarField]> = ScalarCfg::generate_random(L).into_boxed_slice();
      let row: Box<[ScalarField]>= ScalarCfg::generate_random(M).into_boxed_slice();

      // 2) 결과용 버퍼: 반드시 L*M 길이
      let res_template: Box<[G1serde]> = vec![G1serde::zero(); size].into_boxed_slice();

      // 3) G1 generator
      let g1_gen = CurveCfg::generate_random_affine_points(1)[0];

      group.throughput(Throughput::Elements(size as u64));

      // 순차 구현 벤치
      group.bench_with_input(
          BenchmarkId::new("sequential", n),
          &(col.clone(), row.clone(), g1_gen, res_template.clone()),
          |b, (col, row, gen, res)| {
              b.iter(|| {
                  let mut out = res.clone();
                  scaled_outer_product_1d(
                      black_box(col),
                      black_box(row),
                      black_box(&gen),
                      black_box(None),
                      black_box(&mut out),
                  );
                  black_box(out);
              })
          },
      );

      // execute_program 구현 벤치
      group.bench_with_input(
          BenchmarkId::new("icicle_execute", n),
          &(col, row, g1_gen, res_template),
          |b, (col, row, gen, res)| {
              b.iter(|| {
                  let mut out = res.clone();
                  scaled_outer_product_1d_ep(
                      black_box(col),
                      black_box(row),
                      black_box(&gen),
                      black_box(None),
                      black_box(&mut out),
                  );
                  black_box(out);
              })
          },
      );
  }
  group.finish();
}

criterion_group!(benches, bench_scaled_outer);
criterion_main!(benches);