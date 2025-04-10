use ark_bls12_381::{Bls12_381, G1Affine as ArkG1Affine, G2Affine as ArkG2Affine};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::{Field, PrimeField};

// ICICLE 관련 사용 (가정: icicle_bls12_381 크레이트가 제공됨)
use icicle_bls12_381::curve::{
    G1Affine as IcicleG1Affine,
    G2Affine as IcicleG2Affine,
};
use icicle_core::traits::FieldImpl;
use ark_ec::CurveGroup;

pub struct Conversion;

impl Conversion {
  pub fn icicle_g1_affine_to_ark(g: &IcicleG1Affine) -> ArkG1Affine {
      let x_bytes = g.x.to_bytes_le();
      let y_bytes = g.y.to_bytes_le();
      let x = ark_bls12_381::Fq::from_random_bytes(&x_bytes)
          .expect("failed to convert x from icicle to ark");
      let y = ark_bls12_381::Fq::from_random_bytes(&y_bytes)
          .expect("failed to convert y from icicle to ark");
      ArkG1Affine::new_unchecked(x, y)
  }
  
  pub fn icicle_g2_affine_to_ark(g: &IcicleG2Affine) -> ArkG2Affine {
      let x_bytes = g.x.to_bytes_le();
      let y_bytes = g.y.to_bytes_le();
      let x = ark_bls12_381::Fq2::from_random_bytes(&x_bytes)
          .expect("failed to convert x from icicle to ark");
      let y = ark_bls12_381::Fq2::from_random_bytes(&y_bytes)
          .expect("failed to convert y from icicle to ark");
      ArkG2Affine::new_unchecked(x, y)
  }
  
  pub fn verify_bilinearity(g1: ArkG1Affine, g2: ArkG2Affine) -> bool {
      let two = ark_bls12_381::Fr::from(2u64);
      
      let pairing_result = Bls12_381::pairing(g1, g2);
      
      
      let g1_double = (g1.into_group() * two).into_affine();
      let pairing_double = Bls12_381::pairing(g1_double, g2);
      let pairing_squared = pairing_result.0.pow(two.into_bigint());    

      pairing_double.0 == pairing_squared
  }

}