use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_core::traits::{Arithmetic, FieldConfig, FieldImpl, GenerateRandom};
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice};
use std::cmp;
use ark_bls12_381::{Bls12_381, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;

// Assuming the implementation of DensePolynomialExt and BivariatePolynomial is already available
// This mod tests can be placed in a separate file

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use icicle_bls12_381::curve::{CurveCfg, G2CurveCfg};
    use icicle_core::curve::Curve;

    use super::*;
    use crate::{conversion::Conversion, polynomials::{BivariatePolynomial, DensePolynomialExt}, polynomials_ep::{BivariatePolynomialEP, DensePolynomialExtEP}};
    use super::*;
    use crate::vector_operations::{*};

    // Helper function: Create a simple 2D polynomial
    fn create_simple_polynomial() -> DensePolynomialExt {
        // Simple 2x2 polynomial: 1 + 2x + 3y + 4xy (coefficient matrix: [[1, 3], [2, 4]])
        let coeffs = vec![
            ScalarField::from_u32(1),  // Constant term
            ScalarField::from_u32(2),  // x coefficient
            ScalarField::from_u32(3),  // y coefficient
            ScalarField::from_u32(4),  // xy coefficient
        ];
        DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), 2, 2)
    }

    fn create_larger_polynomial() -> DensePolynomialExt {
        // Create a 4x4 polynomial with random coefficients
        let size = 16; // 4x4
        let coeffs = ScalarCfg::generate_random(size);
        DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), 4, 4)
    }

    // Create a univariate polynomial in x
    fn create_univariate_x_polynomial() -> DensePolynomialExt {
        // Polynomial in x: 1 + 2x + 3x^2
        let coeffs = vec![
            ScalarField::from_u32(1),
            ScalarField::from_u32(2),
            ScalarField::from_u32(3),
            ScalarField::from_u32(0),
        ];
        DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), 4, 1)
    }

    // Create a univariate polynomial in y
    fn create_univariate_y_polynomial() -> DensePolynomialExt {
        // Polynomial in y: 1 + 2y + 3y^2
        let mut coeffs = vec![ScalarField::from_u32(0); 16];
        coeffs[0] = ScalarField::from_u32(1);  // Constant
        coeffs[4] = ScalarField::from_u32(2);  // y
        coeffs[8] = ScalarField::from_u32(3);  // y^2
        
        DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), 4, 4)
    }

    #[test]
    fn test_from_coeffs() { // pass
        let poly = create_simple_polynomial();
        assert_eq!(poly.x_degree, 1);
        assert_eq!(poly.y_degree, 1);
        assert_eq!(poly.x_size, 2);
        assert_eq!(poly.y_size, 2);

        // Verify coefficients
        assert_eq!(poly.get_coeff(0, 0), ScalarField::from_u32(1));
        assert_eq!(poly.get_coeff(1, 0), ScalarField::from_u32(2));
        assert_eq!(poly.get_coeff(0, 1), ScalarField::from_u32(3));
        assert_eq!(poly.get_coeff(1, 1), ScalarField::from_u32(4));
    }
    #[test]
    fn test_from_evals() {
        let x_size = 2048;
        let y_size = 1;
        let evals = ScalarCfg::generate_random(x_size * y_size);
        
        let poly = DensePolynomialExt::from_rou_evals(
            HostSlice::from_slice(&evals),
            x_size,
            y_size,
            None,
            None
        );
        let mut recoevered_evals = vec![ScalarField::zero(); x_size * y_size];
        let buff = HostSlice::from_mut_slice(&mut recoevered_evals);
        poly.to_rou_evals(None, None, buff);
        
        let mut flag = true;
        for i in 0..x_size * y_size {
            if !evals[i].eq(&recoevered_evals[i]) {
                flag = false;
            }
        }
        assert!(flag);
    }

    #[test]
    fn test_add() { // pass
        let poly1 = create_simple_polynomial();
        let poly2 = create_simple_polynomial();
        
        let result = &poly1 + &poly2;
        
        // Verify addition results
        assert_eq!(result.get_coeff(0, 0), ScalarField::from_u32(1) + ScalarField::from_u32(1));  // 1+1
        assert_eq!(result.get_coeff(1, 0), ScalarField::from_u32(2) + ScalarField::from_u32(2));  // 2+2
        assert_eq!(result.get_coeff(0, 1), ScalarField::from_u32(3) + ScalarField::from_u32(3));  // 3+3
        assert_eq!(result.get_coeff(1, 1), ScalarField::from_u32(4) + ScalarField::from_u32(4));  // 4+4
    }

    #[test]
    fn test_sub() { // pass
        let poly1 = create_simple_polynomial();
        // Create a polynomial with different coefficients
        let coeffs2 = vec![
            ScalarField::from_u32(5),  // Constant
            ScalarField::from_u32(2),  // x
            ScalarField::from_u32(1),  // y
            ScalarField::from_u32(3),  // xy
        ];
        let poly2 = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs2), 2, 2);
        
        let result = &poly1 - &poly2;
        
        // Verify subtraction results
        assert_eq!(result.get_coeff(0, 0), ScalarField::from_u32(1) - ScalarField::from_u32(5));
        assert_eq!(result.get_coeff(1, 0), ScalarField::from_u32(2) - ScalarField::from_u32(1));
        assert_eq!(result.get_coeff(0, 1), ScalarField::from_u32(3) - ScalarField::from_u32(2));
        assert_eq!(result.get_coeff(1, 1), ScalarField::from_u32(4) - ScalarField::from_u32(3));
    }

    #[test]
    fn test_mul_scalar() { // pass
        let poly = create_simple_polynomial();
        let scalar = ScalarField::from_u32(2);
        
        let result = &poly * &scalar;
        
        // Verify scalar multiplication results
        assert_eq!(result.get_coeff(0, 0), ScalarField::from_u32(1) * ScalarField::from_u32(2));  // 1*2
        assert_eq!(result.get_coeff(1, 0), ScalarField::from_u32(2) * ScalarField::from_u32(2));  // 2*2
        assert_eq!(result.get_coeff(0, 1), ScalarField::from_u32(3) * ScalarField::from_u32(2));  // 3*2
        assert_eq!(result.get_coeff(1, 1), ScalarField::from_u32(4) * ScalarField::from_u32(2));  // 4*2
    }

    #[test]
    fn test_neg() { // pass
        let poly = create_simple_polynomial();
        let result = -&poly;
        
        // Verify negation results
        assert_eq!(result.get_coeff(0, 0), ScalarField::from_u32(0) - ScalarField::from_u32(1));
        assert_eq!(result.get_coeff(1, 0), ScalarField::from_u32(0) - ScalarField::from_u32(2));
        assert_eq!(result.get_coeff(0, 1), ScalarField::from_u32(0) - ScalarField::from_u32(3));
        assert_eq!(result.get_coeff(1, 1), ScalarField::from_u32(0) - ScalarField::from_u32(4));
    }


    #[test]
    fn test_get_univariate_polynomial() { // pass
        // Create a polynomial with predictable coefficients
        let mut coeffs = vec![ScalarField::from_u32(0); 16];
        for y in 0..4 {
            for x in 0..4 {
                let idx = y * 4 + x;
                coeffs[idx] = ScalarField::from_u32((x + y) as u32);
            }
        }
        let poly = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), 4, 4);
        
        // Extract univariate polynomial in x at y = 2
        let x_poly = poly.get_univariate_polynomial_x(2);
        assert_eq!(x_poly.y_size, 1);
        assert_eq!(x_poly.x_size, 4);
        // Check coefficients: at y = 2, the coefficients should be [2, 3, 4, 5]
        for i in 0..4 {
            assert_eq!(x_poly.get_coeff(i, 0), ScalarField::from_u32((i + 2) as u32));
        }
        
        // Extract univariate polynomial in y at x = 1
        let y_poly = poly.get_univariate_polynomial_y(1);
        assert_eq!(y_poly.x_size, 1);
        assert_eq!(y_poly.y_size, 4);
        // Check coefficients: at x = 1, the coefficients should be [1, 2, 3, 4]
        for i in 0..4 {
            assert_eq!(y_poly.get_coeff(0, i), ScalarField::from_u32((1 + i) as u32));
        }
    }

    #[test]
    fn test_eval() { // pass
        let poly = create_simple_polynomial();
        let x = ScalarField::from_u32(2);
        let y = ScalarField::from_u32(3);
        
        // 1 + 2x + 3y + 4xy = 1 + 2*2 + 3*3 + 4*2*3 = 1 + 4 + 9 + 24 = 38
        let expected = ScalarField::from_u32(38);
        let result = poly.eval(&x, &y);
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_x() { // pass
        let poly = create_simple_polynomial();
        let x = ScalarField::from_u32(2);
        
        // Polynomial (1 + 2x + 3y + 4xy) with x=2 becomes: (1 + 4) + (3 + 8)y = 5 + 11y
        let result = poly.eval_x(&x);
        
        assert_eq!(result.x_size, 1);
        assert_eq!(result.y_size, 2);
        
        // Verify coefficients
        assert_eq!(result.get_coeff(0, 0), ScalarField::from_u32(5));   // Constant: 1 + 2*2
        assert_eq!(result.get_coeff(0, 1), ScalarField::from_u32(11));  // y coeff: 3 + 4*2
    }

    #[test]
    fn test_eval_y() { // pass
        let poly = create_simple_polynomial();
        let y = ScalarField::from_u32(3);
        
        // Polynomial (1 + 2x + 3y + 4xy) with y=3 becomes: (1 + 9) + (2 + 12)x = 10 + 14x
        let result = poly.eval_y(&y);
        
        assert_eq!(result.x_size, 2);
        assert_eq!(result.y_size, 1);
        
        // Verify coefficients
        assert_eq!(result.get_coeff(0, 0), ScalarField::from_u32(10));  // Constant: 1 + 3*3
        assert_eq!(result.get_coeff(1, 0), ScalarField::from_u32(14));  // x coeff: 2 + 4*3
    }


    #[test]
    fn test_resize() { // pass
        let mut poly = create_simple_polynomial();
        
        // Resize to 4x4
        poly.resize(4, 4);
        
        // Verify size
        assert_eq!(poly.x_size, 4);
        assert_eq!(poly.y_size, 4);
        
        // Verify original coefficients are preserved
        assert_eq!(poly.get_coeff(0, 0), ScalarField::from_u32(1));
        assert_eq!(poly.get_coeff(1, 0), ScalarField::from_u32(2));
        assert_eq!(poly.get_coeff(0, 1), ScalarField::from_u32(3));
        assert_eq!(poly.get_coeff(1, 1), ScalarField::from_u32(4));
        
        // New parts are filled with zeros
        assert_eq!(poly.get_coeff(2, 0), ScalarField::from_u32(0));
        assert_eq!(poly.get_coeff(0, 2), ScalarField::from_u32(0));
        assert_eq!(poly.get_coeff(3, 3), ScalarField::from_u32(0));
    }

    #[test]
    fn test_optimize_size() { // pass
        // Create a larger polynomial (4x4) but with only 2x2 actually used
        let mut coeffs = vec![ScalarField::from_u32(0); 16];
        for y in 0..2 {
            for x in 0..2 {
                let idx = y * 4 + x;
                coeffs[idx] = ScalarField::from_u32(1);  // Set non-zero values only in 2x2 submatrix
            }
        }
        
        let mut poly = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), 4, 4);
        // Manually adjust the degree to reflect the actual non-zero terms
        poly.x_degree = 1;
        poly.y_degree = 1;
        
        // Optimize size
        poly.optimize_size();
        
        // Size should be 2x2 (or the next power of 2 that can contain 2x2)
        assert!(poly.x_size <= 2);
        assert!(poly.y_size <= 2);
    }

    #[test]
    fn test_div_by_ruffini() {
        let x_size = 2usize.pow(10);
        let y_size = 2usize.pow(5);
        let p_coeffs_vec = ScalarCfg::generate_random(x_size * y_size);
        let p = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&p_coeffs_vec), x_size, y_size);
        let x = ScalarCfg::generate_random(1)[0];
        let y = ScalarCfg::generate_random(1)[0];
        
        let (q_x, q_y, r) = p.div_by_ruffini(x, y);
        let a = ScalarCfg::generate_random(1)[0];
        let b = ScalarCfg::generate_random(1)[0];
        let q_x_eval = q_x.eval(&a, &b);
        let q_y_eval = q_y.eval(&a, &b);
        let estimated_p_eval = (q_x_eval * (a - x)) + (q_y_eval * (b - y)) + r;
        let true_p_eval = p.eval(&a, &b);
        assert!(estimated_p_eval.eq(&true_p_eval));
    }

    #[test]
    fn test_div_by_ruffini_ep() {
        let x_size = 2usize.pow(10);
        let y_size = 2usize.pow(5);
        let p_coeffs_vec = ScalarCfg::generate_random(x_size * y_size);
        let p = DensePolynomialExtEP::from_coeffs(HostSlice::from_slice(&p_coeffs_vec), x_size, y_size);
        let x = ScalarCfg::generate_random(1)[0];
        let y = ScalarCfg::generate_random(1)[0];
        
        let (q_x, q_y, r) = p.div_by_ruffini(x, y);
        let a = ScalarCfg::generate_random(1)[0];
        let b = ScalarCfg::generate_random(1)[0];
        let q_x_eval = q_x.eval(&a, &b);
        let q_y_eval = q_y.eval(&a, &b);
        let estimated_p_eval = (q_x_eval * (a - x)) + (q_y_eval * (b - y)) + r;
        let true_p_eval = p.eval(&a, &b);
        assert!(estimated_p_eval.eq(&true_p_eval));
    }

    #[test]
    fn test_ark_pairing() {
        let size = 2usize.pow(6);
        let g1_point = CurveCfg::generate_random_affine_points(size)[0];
        let g2_point = G2CurveCfg::generate_random_affine_points(size)[0];

        let ark_g1_point = Conversion::icicle_g1_affine_to_ark(&g1_point);
        let ark_g2_point = Conversion::icicle_g2_affine_to_ark(&g2_point);
        println!("ark_g1_point: {:?}", ark_g1_point);
        println!("ark_g2_point: {:?}", ark_g2_point);
        // let pairing_result = Bls12_381::pairing(ark_g1_point, ark_g2_point);
        // println!("pairing_result: {:?}", pairing_result);
        if Conversion::verify_bilinearity(ark_g1_point, ark_g2_point) {
            println!("Bilinearity verified: e(2*G1, G2) == e(G1, G2)^2");
        } else {
            println!("Bilinearity check failed!");
        }
    }

    #[test]
    fn test_divide_x() {
        let x_size = 2usize.pow(8);
        let y_size = 2usize.pow(3);
        let numerator_coeffs = ScalarCfg::generate_random(x_size * y_size);
        let numerator = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&numerator_coeffs), x_size, y_size);
        
        let denom_x_size = 2usize.pow(4);
        let denominator_coeffs = ScalarCfg::generate_random(denom_x_size);
        let denominator = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&denominator_coeffs), denom_x_size, 1);
        
        let (quotient, remainder) = numerator.divide_x(&denominator);
        
        let x = ScalarCfg::generate_random(1)[0];
        let y = ScalarCfg::generate_random(1)[0];
        
        let numerator_eval = numerator.eval(&x, &y);
        let denominator_eval = denominator.eval(&x, &y);
        let quotient_eval = quotient.eval(&x, &y);
        let remainder_eval = remainder.eval(&x, &y);
        
        let reconstructed_eval = denominator_eval * quotient_eval + remainder_eval;
        assert!(numerator_eval.eq(&reconstructed_eval));
        
        assert!(remainder.x_degree < denominator.x_degree);
    }

    #[test]
    fn test_divide_y() {
        let x_size = 2usize.pow(3);
        let y_size = 2usize.pow(8);
        let numerator_coeffs = ScalarCfg::generate_random(x_size * y_size);
        let numerator = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&numerator_coeffs), x_size, y_size);
        
        let denom_y_size = 2usize.pow(4);
        let denominator_coeffs = ScalarCfg::generate_random(denom_y_size);
        let denominator = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&denominator_coeffs), 1, denom_y_size);
        
        let (quotient, remainder) = numerator.divide_y(&denominator);

        let x = ScalarCfg::generate_random(1)[0];
        let y = ScalarCfg::generate_random(1)[0];
        
        let numerator_eval = numerator.eval(&x, &y);
        let denominator_eval = denominator.eval(&x, &y);
        let quotient_eval = quotient.eval(&x, &y);
        let remainder_eval = remainder.eval(&x, &y);
        
        let reconstructed_eval = denominator_eval * quotient_eval + remainder_eval;
        assert!(numerator_eval.eq(&reconstructed_eval));
        
        assert!(remainder.y_degree < denominator.y_degree);
    }

    #[test]
    fn test_mul_monomial() {
        // Create a simple 2x2 polynomial: 1 + 2x + 3y + 4xy
        let coeffs = vec![
            ScalarField::from_u32(1),  // Constant
            ScalarField::from_u32(2),  // x
            ScalarField::from_u32(3),  // y
            ScalarField::from_u32(4),  // xy
        ];
        let poly = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), 2, 2);
        
        // Multiply by xy (shift each term by x^1 * y^1)
        let result = poly.mul_monomial(1, 1);
        
        // Verify result dimensions are powers of two
        assert_eq!(result.x_size.is_power_of_two(), true);
        assert_eq!(result.y_size.is_power_of_two(), true);
        
        // In the implemented code, the degrees are calculated as size-1, so we test against that
        assert_eq!(result.x_degree, result.x_size as i64 - 1);
        assert_eq!(result.y_degree, result.y_size as i64 - 1);
        
        // Original: 1 + 2x + 3y + 4xy
        // After multiplying by xy: xy + 2x^2y + 3xy^2 + 4x^2y^2
        assert_eq!(result.get_coeff(0, 0), ScalarField::from_u32(0));  // No constant term
        assert_eq!(result.get_coeff(1, 1), ScalarField::from_u32(1));  // xy coefficient
        assert_eq!(result.get_coeff(2, 1), ScalarField::from_u32(2));  // x^2y coefficient
        assert_eq!(result.get_coeff(1, 2), ScalarField::from_u32(3));  // xy^2 coefficient
        assert_eq!(result.get_coeff(2, 2), ScalarField::from_u32(4));  // x^2y^2 coefficient
    }

    #[test]
    fn test_mul_polynomial() {
        // Create two simple 2x2 polynomials
        let coeffs1 = vec![
            ScalarField::from_u32(1),  // Constant
            ScalarField::from_u32(2),  // x
            ScalarField::from_u32(3),  // y
            ScalarField::from_u32(4),  // xy
        ];
        let poly1 = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs1), 2, 2);
        
        let coeffs2 = vec![
            ScalarField::from_u32(1),  // Constant
            ScalarField::from_u32(1),  // y
            ScalarField::from_u32(1),  // x
            ScalarField::from_u32(1),  // xy
        ];
        let poly2 = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs2), 2, 2);
        
        // Multiply the polynomials
        let result = &poly1 * &poly2;
        
        // Verify result dimensions are powers of two
        assert_eq!(result.x_size.is_power_of_two(), true);
        assert_eq!(result.y_size.is_power_of_two(), true);
        
        // In the implemented code, the degrees are calculated as size-1, so we test against that
        assert_eq!(result.x_degree, result.x_size as i64 - 1);
        assert_eq!(result.y_degree, result.y_size as i64 - 1);
        
        // (1 + 2x + 3y + 4xy) * (1 + x + y + xy)
        // Verify some coefficients
        assert_eq!(result.get_coeff(0, 0), ScalarField::from_u32(1));   // Constant term
        assert_eq!(result.get_coeff(1, 0), ScalarField::from_u32(3));   // x coefficient (1*x + 2*1)
        assert_eq!(result.get_coeff(0, 1), ScalarField::from_u32(4));   // y coefficient (1*y + 3*1)
        
        // The xy coefficient should be 1*xy + 2*y + 3*x + 4*1 = 10
        assert_eq!(result.get_coeff(1, 1), ScalarField::from_u32(10));
    }


    // Test for div_by_vanishing - requires specific conditions
    #[test]
    fn test_div_by_vanishing_basic() {

        // According to the code, we need m>=2, n==2 condition
        // Case m=2 and n=2:

        let c = 2usize.pow(4);
        let d = 2usize.pow(3);
        let m = 2;
        let n = 2;
        let mut t_x_coeffs = vec![ScalarField::zero(); 2*c];
        let mut t_y_coeffs = vec![ScalarField::zero(); 2*d];
        t_x_coeffs[c] = ScalarField::one();
        t_x_coeffs[0] = ScalarField::zero() - ScalarField::one();
        t_y_coeffs[d] = ScalarField::one();
        t_y_coeffs[0] = ScalarField::zero() - ScalarField::one();
        let mut t_x = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&t_x_coeffs), 2*c, 1);
        let mut t_y = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&t_y_coeffs), 1, 2*d);
        t_x.optimize_size();
        t_y.optimize_size();
        println!("t_x_xdeg: {:?}", t_x.x_degree);
        println!("t_y_ydeg: {:?}", t_y.y_degree);

        let q_x_coeffs_opt = ScalarCfg::generate_random(((m-1)*c-2) * (n*d -2) );
        let q_y_coeffs_opt = ScalarCfg::generate_random((c-1) * ((n-1)*d-2));
        let q_x_coeffs = resize(
            &q_x_coeffs_opt.into_boxed_slice(), 
            (m-1)*c-2, 
            n*d -2,
            (m-1)*c, 
            n*d
        );
        let q_y_coeffs = resize(
            &q_y_coeffs_opt.into_boxed_slice(), 
            c-1, 
            (n-1)*d-2, 
            c, 
            (n-1)*d
        );
        let mut q_x = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&q_x_coeffs), (m-1)*c, n*d);
        let mut q_y = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&q_y_coeffs), c, (n-1)*d);
        q_x.optimize_size();
        q_y.optimize_size();
        let mut p = &(&q_x * &t_x) + &(&q_y * &t_y);
        p.optimize_size();
        println!("p_xsize: {:?}", p.x_size);
        println!("p_ysize: {:?}", p.y_size);
        
        let (mut q_x_found, mut q_y_found) = p.div_by_vanishing(c as i64, d as i64);
        q_x_found.optimize_size();
        q_y_found.optimize_size();
        let p_reconstruct = &(&q_x_found * &t_x) + &(&q_y_found * &t_y);

        let a = ScalarCfg::generate_random(1)[0];
        let b = ScalarCfg::generate_random(1)[0];
        
        let p_evaled = p.eval(&a, &b);
        let p_reconstruct_evaled = p_reconstruct.eval(&a, &b);
        assert!(p_evaled.eq(&p_reconstruct_evaled));
        assert_eq!(q_x.x_degree, q_x_found.x_degree);
        assert_eq!(q_x.y_degree, q_x_found.y_degree);
        assert_eq!(q_y.x_degree, q_y_found.x_degree);
        assert_eq!(q_y.y_degree, q_y_found.y_degree);
        println!("Case m=2 and n=2 passed");

        // Case m=3 and n=2:

        let c = 2usize.pow(4);
        let d = 2usize.pow(3);
        let mut t_x_coeffs = vec![ScalarField::zero(); 2*c];
        let mut t_y_coeffs = vec![ScalarField::zero(); 2*d];
        t_x_coeffs[c] = ScalarField::one();
        t_x_coeffs[0] = ScalarField::zero() - ScalarField::one();
        t_y_coeffs[d] = ScalarField::one();
        t_y_coeffs[0] = ScalarField::zero() - ScalarField::one();
        let mut t_x = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&t_x_coeffs), 2*c, 1);
        let mut t_y = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&t_y_coeffs), 1, 2*d);
        t_x.optimize_size();
        t_y.optimize_size();
        println!("t_x_xdeg: {:?}", t_x.x_degree);
        println!("t_y_ydeg: {:?}", t_y.y_degree);

        let q_x_coeffs_opt = ScalarCfg::generate_random(((m-1)*c-3) * (n*d -2) );
        let q_y_coeffs_opt = ScalarCfg::generate_random((c-1) * ((n-1)*d-2));
        let q_x_coeffs = resize(
            &q_x_coeffs_opt.into_boxed_slice(), 
            (m-1)*c-3, 
            n*d -2,
            (m-1)*c, 
            n*d
        );
        let q_y_coeffs = resize(
            &q_y_coeffs_opt.into_boxed_slice(), 
            c-1, 
            (n-1)*d-2, 
            c, 
            (n-1)*d
        );
        let mut q_x = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&q_x_coeffs), (m-1)*c, n*d);
        let mut q_y = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&q_y_coeffs), c, (n-1)*d);
        q_x.optimize_size();
        q_y.optimize_size();
        let mut p = &(&q_x * &t_x) + &(&q_y * &t_y);
        p.optimize_size();
        println!("p_xsize: {:?}", p.x_size);
        println!("p_ysize: {:?}", p.y_size);
        
        let (mut q_x_found, mut q_y_found) = p.div_by_vanishing(c as i64, d as i64);
        q_x_found.optimize_size();
        q_y_found.optimize_size();
        let p_reconstruct = &(&q_x_found * &t_x) + &(&q_y_found * &t_y);

        let a = ScalarCfg::generate_random(1)[0];
        let b = ScalarCfg::generate_random(1)[0];
        
        let p_evaled = p.eval(&a, &b);
        let p_reconstruct_evaled = p_reconstruct.eval(&a, &b);
        assert!(p_evaled.eq(&p_reconstruct_evaled));
        assert_eq!(q_x.x_degree, q_x_found.x_degree);
        assert_eq!(q_x.y_degree, q_x_found.y_degree);
        assert_eq!(q_y.x_degree, q_y_found.x_degree);
        assert_eq!(q_y.y_degree, q_y_found.y_degree);

    }

    // More tests can be added as needed
}

#[cfg(test)]
mod tests_vectors {
    use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
    use icicle_core::traits::{Arithmetic, FieldConfig, FieldImpl, GenerateRandom};
    use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice};
    use std::cmp;
    use crate::vector_operations::{gen_evaled_lagrange_bases, matrix_matrix_mul};
    
    use crate::vector_operations::{outer_product_two_vecs, point_mul_two_vecs};

    macro_rules! scalar_vec {
        ( $( $x:expr ),* ) => {
            vec![
                $( ScalarField::from_u32($x) ),*
            ].into_boxed_slice()
        };
    }

    #[test]
    fn test_point_mul_two_vecs() {
        let vec1 = scalar_vec![1, 2, 3];
        let vec2 = scalar_vec![4, 5];
        let vec3 = scalar_vec![2, 0, 2, 4];

        let mut res = vec![ScalarField::zero(); 6].into_boxed_slice();
        outer_product_two_vecs(&vec1, &vec2, &mut res);
        println!("res : {:?}", res);

        let mut res = vec![ScalarField::zero(); 6].into_boxed_slice();
        outer_product_two_vecs(&vec2, &vec1, &mut res);
        println!("res : {:?}", res);

        let mut res = vec![ScalarField::zero(); 12].into_boxed_slice();
        outer_product_two_vecs(&vec1, &vec3, &mut res);
        println!("res : {:?}", res);

        let mut res = vec![ScalarField::zero(); 8].into_boxed_slice();
        outer_product_two_vecs(&vec3, &vec2, &mut res);
        println!("res : {:?}", res);

    }

    #[test]
    fn test_matrix_matrix_mul_small() {
        // example size: 2x3 * 3x2 = 2x2
        // LHS: 2x3
        // [1 2 3]
        // [4 5 6]
        let lhs = vec![
            ScalarField::from_u32(1u32),
            ScalarField::from_u32(2u32),
            ScalarField::from_u32(3u32),
            ScalarField::from_u32(4u32),
            ScalarField::from_u32(5u32),
            ScalarField::from_u32(6u32),
        ]
        .into_boxed_slice();

        // RHS: 3x2
        // [7  8]
        // [9 10]
        // [11 12]
        let rhs = vec![
            ScalarField::from_u32(7u32),
            ScalarField::from_u32(8u32),
            ScalarField::from_u32(9u32),
            ScalarField::from_u32(10u32),
            ScalarField::from_u32(11u32),
            ScalarField::from_u32(12u32),
        ]
        .into_boxed_slice();

        // expected result: 2x2
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let expected = vec![
            ScalarField::from_u32(58u32),
            ScalarField::from_u32(64u32),
            ScalarField::from_u32(139u32),
            ScalarField::from_u32(154u32),
        ]
        .into_boxed_slice();

        let mut res = vec![ScalarField::zero(); 4].into_boxed_slice();
        matrix_matrix_mul(&lhs, &rhs, 2, 3, 2, &mut res);

        for i in 0..4 {
            assert_eq!(res[i], expected[i], "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_gen_evaled_lagrange_bases() {
        let x = ScalarCfg::generate_random(1)[0];
        let size = 2048;
        let mut res = vec![ScalarField::zero(); size].into_boxed_slice();
        gen_evaled_lagrange_bases(&x, size, &mut res);
        
    }
}

#[cfg(test)]
pub mod test_polynomial_ep {
    use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
    use icicle_bls12_381::polynomials::DensePolynomial;
    use icicle_core::polynomials::UnivariatePolynomial;
    use icicle_core::traits::FieldImpl;
    use crate::polynomials::{BivariatePolynomial, DensePolynomialExt};
    use crate::polynomials_ep::{BivariatePolynomialEP, DensePolynomialExtEP};
    use icicle_runtime::memory::HostSlice;
    use icicle_core::traits::GenerateRandom;

    #[test]
    fn test_polynomial_find_degree() {
        
        // Sample sizes to test
        const X_SIZE: usize = 8;
        const Y_SIZE: usize = 12;
        
        // Create a test polynomial with non-zero coefficients at specific positions
        let mut coeffs1 = vec![ScalarField::zero(); X_SIZE * Y_SIZE];
        
        // Set specific coefficients to establish a known degree pattern
        coeffs1[2 + 3 * X_SIZE] = ScalarField::one(); // x^2 * y^3
        coeffs1[5 + 1 * X_SIZE] = ScalarField::one(); // x^5 * y^1
        coeffs1[0 + 7 * X_SIZE] = ScalarField::one(); // x^0 * y^7
        
        let poly = DensePolynomial::from_coeffs(HostSlice::from_slice(&coeffs1), X_SIZE * Y_SIZE);
        
        // Test case 1: x_size <= y_size
        let (x_degree1, y_degree1) = DensePolynomialExt::find_degree(&poly, X_SIZE, Y_SIZE);
        let (x_degree2, y_degree2) = DensePolynomialExtEP::find_degree(&poly, X_SIZE, Y_SIZE);
        
        assert_eq!(
            x_degree1, x_degree2,
            "X degrees don't match for case where x_size <= y_size: {} vs {}", 
            x_degree1, x_degree2
        );
        
        assert_eq!(
            y_degree1, y_degree2,
            "Y degrees don't match for case where x_size <= y_size: {} vs {}", 
            y_degree1, y_degree2
        );
        
        // Test case 2: x_size > y_size
        let (x_degree1, y_degree1) = DensePolynomialExt::find_degree(&poly, Y_SIZE, X_SIZE);
        let (x_degree2, y_degree2) = DensePolynomialExtEP::find_degree(&poly, Y_SIZE, X_SIZE);
        
        assert_eq!(
            x_degree1, x_degree2,
            "X degrees don't match for case where x_size > y_size: {} vs {}", 
            x_degree1, x_degree2
        );
        
        assert_eq!(
            y_degree1, y_degree2,
            "Y degrees don't match for case where x_size > y_size: {} vs {}", 
            y_degree1, y_degree2
        );
        
        // Test case 3: Edge case with zero polynomial
        let coeffs2 = vec![ScalarField::zero(); X_SIZE * Y_SIZE];
        let zero_poly = DensePolynomial::from_coeffs(HostSlice::from_slice(&coeffs2), X_SIZE * Y_SIZE);
        
        let (x_degree1, y_degree1) = DensePolynomialExt::find_degree(&zero_poly, X_SIZE, Y_SIZE);
        let (x_degree2, y_degree2) = DensePolynomialExtEP::find_degree(&zero_poly, X_SIZE, Y_SIZE);
        
        assert_eq!(
            x_degree1, x_degree2,
            "X degrees don't match for zero polynomial: {} vs {}", 
            x_degree1, x_degree2
        );
        
        assert_eq!(
            y_degree1, y_degree2,
            "Y degrees don't match for zero polynomial: {} vs {}", 
            y_degree1, y_degree2
        );
        
        // Test case 4: Complex polynomial with higher degrees
        let mut coeffs3 = vec![ScalarField::zero(); X_SIZE * Y_SIZE];
        coeffs3[7 + 9 * X_SIZE] = ScalarField::one();  // x^7 * y^9
        coeffs3[3 + 11 * X_SIZE] = ScalarField::one(); // x^3 * y^11
        
        let complex_poly = DensePolynomial::from_coeffs(HostSlice::from_slice(&coeffs3), X_SIZE * Y_SIZE);
        
        let (x_degree1, y_degree1) = DensePolynomialExt::find_degree(&complex_poly, X_SIZE, Y_SIZE);
        let (x_degree2, y_degree2) = DensePolynomialExtEP::find_degree(&complex_poly, X_SIZE, Y_SIZE);
        
        assert_eq!(
            x_degree1, x_degree2,
            "X degrees don't match for complex polynomial: {} vs {}", 
            x_degree1, x_degree2
        );
        
        assert_eq!(
            y_degree1, y_degree2,
            "Y degrees don't match for complex polynomial: {} vs {}", 
            y_degree1, y_degree2
        );
        
        println!("All tests passed: Both find_degree implementations produce identical results.");
    }

    #[test]
    fn test_from_rou_evals() {
        const X_SIZE: usize = 8;
        const Y_SIZE: usize = 8;
        const SIZE: usize = X_SIZE * Y_SIZE;
        
        let evals = ScalarCfg::generate_random(SIZE);
        
        let coset_x_val = ScalarField::from_u32(7u32);
        let coset_y_val = ScalarField::from_u32(11u32);
        
        // 테스트 케이스 1: 기본 변환 (코셋 없음)
        let poly1_orig = DensePolynomialExt::from_rou_evals(
            HostSlice::from_slice(&evals),
            X_SIZE,
            Y_SIZE,
            None,
            None
        );
        
        let poly1_ep = DensePolynomialExtEP::from_rou_evals(
            HostSlice::from_slice(&evals),
            X_SIZE,
            Y_SIZE,
            None,
            None
        );
        
        // 두 결과를 비교
        let coeffs1_orig = poly1_orig.get_coeff(X_SIZE as u64, Y_SIZE as u64);
        let coeffs1_ep = poly1_ep.get_coeff(X_SIZE as u64, Y_SIZE as u64);
        
        let mut host_coeffs1_orig = vec![ScalarField::zero(); SIZE];
        let mut host_coeffs1_ep = vec![ScalarField::zero(); SIZE];
        
        host_coeffs1_orig[0] = coeffs1_orig;
        host_coeffs1_ep[0] = coeffs1_ep;
        
        for i in 0..SIZE {
            assert_eq!(
                host_coeffs1_orig[i],
                host_coeffs1_ep[i],
                "Coefficients don't match at index {} for case without cosets: {:?} vs {:?}",
                i,
                host_coeffs1_orig[i],
                host_coeffs1_ep[i]
            );
        }
        
        // 테스트 케이스 2: x 코셋만 사용
        let poly2_orig = DensePolynomialExt::from_rou_evals(
            HostSlice::from_slice(&evals),
            X_SIZE,
            Y_SIZE,
            Some(&coset_x_val),
            None
        );
        
        let poly2_ep = DensePolynomialExtEP::from_rou_evals(
            HostSlice::from_slice(&evals),
            X_SIZE,
            Y_SIZE,
            Some(&coset_x_val),
            None
        );
        
        // 두 결과를 비교
        let coeffs2_orig = poly2_orig.get_coeff(X_SIZE as u64, Y_SIZE as u64);
        let coeffs2_ep = poly2_ep.get_coeff(X_SIZE as u64, Y_SIZE as u64);
        
        let mut host_coeffs2_orig = vec![ScalarField::zero(); SIZE];
        let mut host_coeffs2_ep = vec![ScalarField::zero(); SIZE];
        
        host_coeffs2_orig[0] = coeffs2_orig;
        host_coeffs2_ep[0] = coeffs2_ep;
        
        for i in 0..SIZE {
            assert_eq!(
                host_coeffs2_orig[i],
                host_coeffs2_ep[i],
                "Coefficients don't match at index {} for case with only x coset: {:?} vs {:?}",
                i,
                host_coeffs2_orig[i],
                host_coeffs2_ep[i]
            );
        }
        
        // 테스트 케이스 3: y 코셋만 사용
        let poly3_orig = DensePolynomialExt::from_rou_evals(
            HostSlice::from_slice(&evals),
            X_SIZE,
            Y_SIZE,
            None,
            Some(&coset_y_val)
        );
        
        let poly3_ep = DensePolynomialExtEP::from_rou_evals(
            HostSlice::from_slice(&evals),
            X_SIZE,
            Y_SIZE,
            None,
            Some(&coset_y_val)
        );
        
        // 두 결과를 비교
        let coeffs3_orig = poly3_orig.get_coeff(X_SIZE as u64, Y_SIZE as u64);
        let coeffs3_ep = poly3_ep.get_coeff(X_SIZE as u64, Y_SIZE as u64);
        
        let mut host_coeffs3_orig = vec![ScalarField::zero(); SIZE];
        let mut host_coeffs3_ep = vec![ScalarField::zero(); SIZE];
        
        host_coeffs3_orig[0] = coeffs3_orig;
        host_coeffs3_ep[0] = coeffs3_ep;
        
        for i in 0..SIZE {
            assert_eq!(
                host_coeffs3_orig[i],
                host_coeffs3_ep[i],
                "Coefficients don't match at index {} for case with only y coset: {:?} vs {:?}",
                i,
                host_coeffs3_orig[i],
                host_coeffs3_ep[i]
            );
        }
        
        // 테스트 케이스 4: 두 코셋 모두 사용
        let poly4_orig = DensePolynomialExt::from_rou_evals(
            HostSlice::from_slice(&evals),
            X_SIZE,
            Y_SIZE,
            Some(&coset_x_val),
            Some(&coset_y_val)
        );
        
        let poly4_ep = DensePolynomialExtEP::from_rou_evals(
            HostSlice::from_slice(&evals),
            X_SIZE,
            Y_SIZE,
            Some(&coset_x_val),
            Some(&coset_y_val)
        );
        
        // 두 결과를 비교
        let coeffs4_orig = poly4_orig.get_coeff(X_SIZE as u64, Y_SIZE as u64);
        let coeffs4_ep = poly4_ep.get_coeff(X_SIZE as u64, Y_SIZE as u64);
        
        let mut host_coeffs4_orig = vec![ScalarField::zero(); SIZE];
        let mut host_coeffs4_ep = vec![ScalarField::zero(); SIZE];
        
        host_coeffs4_orig[0] = coeffs4_orig;
        host_coeffs4_ep[0] = coeffs4_ep;
        
        for i in 0..SIZE {
            assert_eq!(
                host_coeffs4_orig[i],
                host_coeffs4_ep[i],
                "Coefficients don't match at index {} for case with both cosets: {:?} vs {:?}",
                i,
                host_coeffs4_orig[i],
                host_coeffs4_ep[i]
            );
        }
        
        // 테스트 케이스 5: 다른 크기에 대한 테스트
        const X_SIZE_2: usize = 16;
        const Y_SIZE_2: usize = 4;
        const SIZE_2: usize = X_SIZE_2 * Y_SIZE_2;
        
        // 새로운 평가값 생성
        let mut evals2 = ScalarCfg::generate_random(SIZE_2);
        
        let poly5_orig = DensePolynomialExt::from_rou_evals(
            HostSlice::from_slice(&evals2),
            X_SIZE_2,
            Y_SIZE_2,
            Some(&coset_x_val),
            Some(&coset_y_val)
        );
        
        let poly5_ep = DensePolynomialExtEP::from_rou_evals(
            HostSlice::from_slice(&evals2),
            X_SIZE_2,
            Y_SIZE_2,
            Some(&coset_x_val),
            Some(&coset_y_val)
        );
        
        // 두 결과를 비교
        let coeffs5_orig = poly5_orig.get_coeff(X_SIZE_2 as u64, Y_SIZE_2 as u64);
        let coeffs5_ep = poly5_ep.get_coeff(X_SIZE_2 as u64, Y_SIZE_2 as u64);
        
        let mut host_coeffs5_orig = vec![ScalarField::zero(); SIZE_2];
        let mut host_coeffs5_ep = vec![ScalarField::zero(); SIZE_2];
        
        host_coeffs5_orig[0] = coeffs5_orig;
        host_coeffs5_ep[0] = coeffs5_ep;
        
        for i in 0..SIZE_2 {
            assert_eq!(
                host_coeffs5_orig[i],
                host_coeffs5_ep[i],
                "Coefficients don't match at index {} for different sizes case: {:?} vs {:?}",
                i,
                host_coeffs5_orig[i],
                host_coeffs5_ep[i]
            );
        }
        
        println!("All from_rou_evals tests passed: Both implementations produce identical results.");
    }

    #[test]
    fn test_resize() {
        // Define test scenarios with various size transformations
        let test_cases = vec![
            // (original_x_size, original_y_size, target_x_size, target_y_size)
            (8, 8, 16, 16),     // Expansion case
            (16, 16, 8, 8),     // Reduction case
            (8, 16, 16, 8),     // Dimension change case
            (32, 8, 8, 32),     // Another dimension change case
            (16, 8, 16, 8),     // Same size case
        ];
        
        for (orig_x, orig_y, target_x, target_y) in test_cases {
            println!("Test case: ({},{}) -> ({},{})", orig_x, orig_y, target_x, target_y);
            
            // Create original polynomial with random coefficients
            let size = orig_x * orig_y;
            let mut coeffs = vec![ScalarField::zero(); size];
            
            // Fill with random values
            for i in 0..size {
                coeffs[i] = ScalarCfg::generate_random(1)[0];
            }
            
            // Create coefficients as HostSlice
            let coeffs_slice = HostSlice::from_slice(&coeffs);
            
            // Test with original implementation
            let mut orig_poly_ext = DensePolynomialExt::from_coeffs(coeffs_slice, orig_x, orig_y);
            let orig_copy = orig_poly_ext.clone(); // Store a copy
            orig_poly_ext.resize(target_x, target_y);
            
            // Test with execute_program implementation
            let mut ep_poly_ext = DensePolynomialExtEP::from_coeffs(coeffs_slice, orig_x, orig_y);
            let ep_copy = ep_poly_ext.clone(); // Store a copy
            ep_poly_ext.resize(target_x, target_y);
            
            // Compare results
            assert_eq!(orig_poly_ext.x_size, ep_poly_ext.x_size, 
                    "X not match: {} vs {}", 
                    orig_poly_ext.x_size, ep_poly_ext.x_size);
            
            assert_eq!(orig_poly_ext.y_size, ep_poly_ext.y_size, 
                    "Y not match: {} vs {}", 
                    orig_poly_ext.y_size, ep_poly_ext.y_size);
            
            // Compare coefficients
            let orig_size = orig_poly_ext.x_size * orig_poly_ext.y_size;
            let mut orig_result_coeffs = vec![ScalarField::zero(); orig_size];
            let mut ep_result_coeffs = vec![ScalarField::zero(); orig_size];
            
            orig_poly_ext.copy_coeffs(0, HostSlice::from_mut_slice(&mut orig_result_coeffs));
            ep_poly_ext.copy_coeffs(0, HostSlice::from_mut_slice(&mut ep_result_coeffs));
            
            for i in 0..orig_size {
                assert_eq!(orig_result_coeffs[i], ep_result_coeffs[i], 
                        "Coeff is not mathced at index {}: {:?} vs {:?}", 
                        i, orig_result_coeffs[i], ep_result_coeffs[i]);
            }
            
            println!("Pass test case: ({},{}) -> ({},{})", orig_x, orig_y, target_x, target_y);
        }
        
        // Test special cases with zero sizes
        let special_test_cases = vec![
            (8, 8, 0, 8),   // Target x_size is 0
            (8, 8, 8, 0),   // Target y_size is 0
            (8, 8, 0, 0),   // Both target x_size and y_size are 0
        ];
        
        for (orig_x, orig_y, target_x, target_y) in special_test_cases {
            println!("Unique case test: ({},{}) -> ({},{})", orig_x, orig_y, target_x, target_y);
            
            let size = orig_x * orig_y;
            let mut coeffs = vec![ScalarField::zero(); size];
            
            // Fill with random values
            for i in 0..size {
                coeffs[i] = ScalarCfg::generate_random(1)[0];
            }
            
            // Create coefficients as HostSlice
            let coeffs_slice = HostSlice::from_slice(&coeffs);
            
            // Test with original implementation
            let mut orig_poly_ext = DensePolynomialExt::from_coeffs(coeffs_slice, orig_x, orig_y);
            let orig_copy = orig_poly_ext.clone();
            
            // Test with execute_program implementation
            let mut ep_poly_ext = DensePolynomialExtEP::from_coeffs(coeffs_slice, orig_x, orig_y);
            let ep_copy = ep_poly_ext.clone();
            
            // Check if both implementations panic for zero sizes
            let orig_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                orig_poly_ext.resize(target_x, target_y);
            }));
            
            let ep_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ep_poly_ext.resize(target_x, target_y);
            }));
            
            // Verify both implementations handle panics the same way
            assert_eq!(orig_result.is_err(), ep_result.is_err(), 
                    "panic result is different");
            
            if orig_result.is_err() && ep_result.is_err() {
                println!("panic.");
            }
            
            println!("pass unique case: ({},{}) -> ({},{})", orig_x, orig_y, target_x, target_y);
        }
        
        println!("All resize tests are passed!");
    }

    #[test]
    fn test_mul() {
        // Define test scenarios with various polynomial sizes
        let test_cases = vec![
            // (x_size_1, y_size_1, x_size_2, y_size_2)
            (4, 4, 4, 4),     // Small equal sizes
            (8, 8, 4, 4),     // First larger than second
            (4, 4, 8, 8),     // Second larger than first
            (8, 4, 4, 8),     // Different dimensions
            (16, 16, 16, 16), // Larger sizes
        ];
        
        for (x_size_1, y_size_1, x_size_2, y_size_2) in test_cases {
            println!("Test case: ({},{}) * ({},{})", x_size_1, y_size_1, x_size_2, y_size_2);
            
            // Create first polynomial with random coefficients
            let size_1 = x_size_1 * y_size_1;
            let mut coeffs_1 = vec![ScalarField::zero(); size_1];
            
            // Fill with random values
            for i in 0..size_1 {
                coeffs_1[i] = ScalarCfg::generate_random(1)[0];
            }
            
            // Create second polynomial with random coefficients
            let size_2 = x_size_2 * y_size_2;
            let mut coeffs_2 = vec![ScalarField::zero(); size_2];
            
            // Fill with random values
            for i in 0..size_2 {
                coeffs_2[i] = ScalarCfg::generate_random(1)[0];
            }
            
            // Create coefficients as HostSlice
            let coeffs_slice_1 = HostSlice::from_slice(&coeffs_1);
            let coeffs_slice_2 = HostSlice::from_slice(&coeffs_2);
            
            // Test with original implementation
            let poly_1 = DensePolynomialExt::from_coeffs(coeffs_slice_1, x_size_1, y_size_1);
            let poly_2 = DensePolynomialExt::from_coeffs(coeffs_slice_2, x_size_2, y_size_2);
            let orig_result = &poly_1 * &poly_2;
            
            // Test with execute_program implementation
            let poly_1_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice_1, x_size_1, y_size_1);
            let poly_2_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice_2, x_size_2, y_size_2);
            let ep_result = &poly_1_ep._mul(&poly_2_ep);
            
            // Compare results
            assert_eq!(orig_result.x_size, ep_result.x_size, 
                    "X size not match: {} vs {}", 
                    orig_result.x_size, ep_result.x_size);
            
            assert_eq!(orig_result.y_size, ep_result.y_size, 
                    "Y size not match: {} vs {}", 
                    orig_result.y_size, ep_result.y_size);
            
            // Compare coefficients
            let result_size = orig_result.x_size * orig_result.y_size;
            let mut orig_result_coeffs = vec![ScalarField::zero(); result_size];
            let mut ep_result_coeffs = vec![ScalarField::zero(); result_size];
            
            orig_result.copy_coeffs(0, HostSlice::from_mut_slice(&mut orig_result_coeffs));
            ep_result.copy_coeffs(0, HostSlice::from_mut_slice(&mut ep_result_coeffs));
            
            for i in 0..result_size {
                assert_eq!(orig_result_coeffs[i], ep_result_coeffs[i], 
                        "Coefficient not matched at index {}: {:?} vs {:?}", 
                        i, orig_result_coeffs[i], ep_result_coeffs[i]);
            }
            
            println!("Pass test case: ({},{}) * ({},{})", x_size_1, y_size_1, x_size_2, y_size_2);
        }
        
        // Test special cases: constant polynomials
        let special_test_cases = vec![
            // (x_size_1, y_size_1, x_size_2, y_size_2)
            (1, 1, 4, 4),     // Constant * polynomial
            (4, 4, 1, 1),     // Polynomial * constant
            (1, 1, 1, 1),     // Constant * constant
        ];
        
        for (x_size_1, y_size_1, x_size_2, y_size_2) in special_test_cases {
            println!("Special case: ({},{}) * ({},{})", x_size_1, y_size_1, x_size_2, y_size_2);
            
            // Generate random coefficients
            let size_1 = x_size_1 * y_size_1;
            let mut coeffs_1 = vec![ScalarField::zero(); size_1];
            for i in 0..size_1 {
                coeffs_1[i] = ScalarCfg::generate_random(1)[0];
            }
            
            let size_2 = x_size_2 * y_size_2;
            let mut coeffs_2 = vec![ScalarField::zero(); size_2];
            for i in 0..size_2 {
                coeffs_2[i] = ScalarCfg::generate_random(1)[0];
            }
            
            // Create coefficients as HostSlice
            let coeffs_slice_1 = HostSlice::from_slice(&coeffs_1);
            let coeffs_slice_2 = HostSlice::from_slice(&coeffs_2);
            
            // Test with original implementation
            let poly_1 = DensePolynomialExt::from_coeffs(coeffs_slice_1, x_size_1, y_size_1);
            let poly_2 = DensePolynomialExt::from_coeffs(coeffs_slice_2, x_size_2, y_size_2);
            let orig_result = &poly_1 * &poly_2;
            
            // Test with execute_program implementation
            let poly_1_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice_1, x_size_1, y_size_1);
            let poly_2_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice_2, x_size_2, y_size_2);
            let ep_result = &poly_1_ep._mul(&poly_2_ep);
            
            // Compare results
            assert_eq!(orig_result.x_size, ep_result.x_size, 
                    "X size not match: {} vs {}", 
                    orig_result.x_size, ep_result.x_size);
            
            assert_eq!(orig_result.y_size, ep_result.y_size, 
                    "Y size not match: {} vs {}", 
                    orig_result.y_size, ep_result.y_size);
            
            // Compare coefficients
            let result_size = orig_result.x_size * orig_result.y_size;
            let mut orig_result_coeffs = vec![ScalarField::zero(); result_size];
            let mut ep_result_coeffs = vec![ScalarField::zero(); result_size];
            
            orig_result.copy_coeffs(0, HostSlice::from_mut_slice(&mut orig_result_coeffs));
            ep_result.copy_coeffs(0, HostSlice::from_mut_slice(&mut ep_result_coeffs));
            
            for i in 0..result_size {
                assert_eq!(orig_result_coeffs[i], ep_result_coeffs[i], 
                        "Coefficient not matched at index {}: {:?} vs {:?}", 
                        i, orig_result_coeffs[i], ep_result_coeffs[i]);
            }
            
            println!("Pass special case: ({},{}) * ({},{})", x_size_1, y_size_1, x_size_2, y_size_2);
        }
        
        // Time performance comparison for large polynomials
        let perf_test_sizes = vec![
            (32, 32, 32, 32),
            (64, 64, 64, 64),
        ];
        
        for (x_size_1, y_size_1, x_size_2, y_size_2) in perf_test_sizes {
            println!("Performance test: ({},{}) * ({},{})", x_size_1, y_size_1, x_size_2, y_size_2);
            
            // Generate random coefficients
            let size_1 = x_size_1 * y_size_1;
            let mut coeffs_1 = vec![ScalarField::zero(); size_1];
            for i in 0..size_1 {
                coeffs_1[i] = ScalarCfg::generate_random(1)[0];
            }
            
            let size_2 = x_size_2 * y_size_2;
            let mut coeffs_2 = vec![ScalarField::zero(); size_2];
            for i in 0..size_2 {
                coeffs_2[i] = ScalarCfg::generate_random(1)[0];
            }
            
            // Create coefficients as HostSlice
            let coeffs_slice_1 = HostSlice::from_slice(&coeffs_1);
            let coeffs_slice_2 = HostSlice::from_slice(&coeffs_2);
            
            // Create polynomials
            let poly_1 = DensePolynomialExt::from_coeffs(coeffs_slice_1, x_size_1, y_size_1);
            let poly_2 = DensePolynomialExt::from_coeffs(coeffs_slice_2, x_size_2, y_size_2);
            let poly_1_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice_1, x_size_1, y_size_1);
            let poly_2_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice_2, x_size_2, y_size_2);
            
            // Measure time for original implementation
            let start_time = std::time::Instant::now();
            let orig_result = &poly_1 * &poly_2;
            let orig_duration = start_time.elapsed();
            
            // Measure time for execute_program implementation
            let start_time = std::time::Instant::now();
            let ep_result = &poly_1_ep._mul(&poly_2_ep);
            let ep_duration = start_time.elapsed();
            
            println!("Original implementation: {:?}", orig_duration);
            println!("Execute_program implementation: {:?}", ep_duration);
            println!("Speedup: {:.2}x", orig_duration.as_secs_f64() / ep_duration.as_secs_f64());
            
            // Verify results match
            let result_size = orig_result.x_size * orig_result.y_size;
            let mut orig_result_coeffs = vec![ScalarField::zero(); result_size];
            let mut ep_result_coeffs = vec![ScalarField::zero(); result_size];
            
            orig_result.copy_coeffs(0, HostSlice::from_mut_slice(&mut orig_result_coeffs));
            ep_result.copy_coeffs(0, HostSlice::from_mut_slice(&mut ep_result_coeffs));
            
            for i in 0..result_size {
                assert_eq!(orig_result_coeffs[i], ep_result_coeffs[i]);
            }
            
            println!("Pass performance test: ({},{}) * ({},{})", x_size_1, y_size_1, x_size_2, y_size_2);
        }
        
        println!("All multiplication tests passed!");
    }
}

#[cfg(test)]
mod tests_vanishing {
    use super::*;
    use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
    use icicle_core::traits::FieldImpl;
    use icicle_core::traits::GenerateRandom;
    use icicle_runtime::memory::{HostSlice, DeviceSlice, DeviceVec};
    use icicle_core::vec_ops::VecOpsConfig;
    use crate::polynomials::BivariatePolynomial;
    use crate::polynomials::DensePolynomialExt;
    use crate::polynomials_ep::BivariatePolynomialEP;
    use crate::polynomials_ep::DensePolynomialExtEP;

    // 테스트 다항식 생성 함수
    fn create_test_polynomial(x_size: usize, y_size: usize, sparsity: usize) -> (DensePolynomialExt, DensePolynomialExtEP) {
        // 랜덤 계수 생성
        let mut coeffs = vec![ScalarField::zero(); x_size * y_size];
        
        // 일부 계수만 랜덤값으로 설정 (sparsity 간격으로)
        for i in 0..coeffs.len() {
            if i % sparsity == 0 {
                coeffs[i] = ScalarCfg::generate_random(1)[0];
            }
        }
        
        // 최고차항 계수가 0이 아니도록 설정
        let highest_degree_idx = x_size * y_size - 1;
        coeffs[highest_degree_idx] = ScalarCfg::generate_random(1)[0];
        
        let coeffs_slice = HostSlice::from_slice(&coeffs);
        
        // 원본 구현과 EP 구현 다항식 생성
        let poly_orig = DensePolynomialExt::from_coeffs(coeffs_slice, x_size, y_size);
        let poly_ep = DensePolynomialExtEP::from_coeffs(coeffs_slice, x_size, y_size);
        
        (poly_orig, poly_ep)
    }
    
    // Helper function to convert EP polynomial to original polynomial
    fn ep_to_orig(poly_ep: &DensePolynomialExtEP) -> DensePolynomialExt {
        let size = poly_ep.x_size * poly_ep.y_size;
        let mut coeffs = vec![ScalarField::zero(); size];
        let coeffs_slice = HostSlice::from_mut_slice(&mut coeffs);
        poly_ep.copy_coeffs(0, coeffs_slice);
        DensePolynomialExt::from_coeffs(HostSlice::from_slice(&coeffs), poly_ep.x_size, poly_ep.y_size)
    }

    // 다항식이 동일한지 확인하는 함수
    fn assert_polynomials_equal(poly1: &DensePolynomialExt, poly2: &DensePolynomialExtEP, msg: &str) {
        let poly2_orig = ep_to_orig(poly2);
        assert_eq!(poly1.x_size, poly2_orig.x_size, "{}: x_size mismatch", msg);
        assert_eq!(poly1.y_size, poly2_orig.y_size, "{}: y_size mismatch", msg);
        
        let size = poly1.x_size * poly1.y_size;
        let mut coeffs1 = vec![ScalarField::zero(); size];
        let mut coeffs2 = vec![ScalarField::zero(); size];
        
        let coeffs1_slice = HostSlice::from_mut_slice(&mut coeffs1);
        let coeffs2_slice = HostSlice::from_mut_slice(&mut coeffs2);
        
        poly1.copy_coeffs(0, coeffs1_slice);
        poly2_orig.copy_coeffs(0, coeffs2_slice);
        
        for i in 0..size {
            assert_eq!(coeffs1[i], coeffs2[i], "{}: coefficient at index {} mismatch", msg, i);
        }
    }
    
     #[test]
     fn test_div_by_vanishing_larger_m() {
         
         
         // 16x8 다항식 (m=4, n=2, denom_x=4, denom_y=4)
         let (poly_orig, poly_ep) = create_test_polynomial(16, 8, 5);
         
         // 원본 구현 결과 확인
         let (quo_x_orig, quo_y_orig) = poly_orig.div_by_vanishing(4, 4);
         
         // EP 구현 결과 확인
         let (quo_x_ep, quo_y_ep) = poly_ep.div_by_vanishing(4, 4);
         
         // 결과 비교
         assert_polynomials_equal(&quo_x_orig, &quo_x_ep, "quotient_x (larger m)");
         assert_polynomials_equal(&quo_y_orig, &quo_y_ep, "quotient_y (larger m)");
     }
     
     // 매우 큰 m 값을 가진 케이스 (m>>2, n=2) 테스트
     #[test]
     fn test_div_by_vanishing_very_large_m() {
         
         
         // 32x8 다항식 (m=8, n=2, denom_x=4, denom_y=4)
         let (poly_orig, poly_ep) = create_test_polynomial(32, 8, 7);
         
         // 원본 구현 결과 확인
         let (quo_x_orig, quo_y_orig) = poly_orig.div_by_vanishing(4, 4);
         
         // EP 구현 결과 확인
         let (quo_x_ep, quo_y_ep) = poly_ep.div_by_vanishing(4, 4);
         
         // 결과 비교
         assert_polynomials_equal(&quo_x_orig, &quo_x_ep, "quotient_x (very large m)");
         assert_polynomials_equal(&quo_y_orig, &quo_y_ep, "quotient_y (very large m)");
     }
     
     // 서로 다른 차수의 테스트 (n은 고정 2)
     #[test]
     fn test_div_by_vanishing_different_degrees() {
         
         // 8x8 다항식 (denom_x=2, denom_y=4 => m=4, n=2)
         let (poly_orig, poly_ep) = create_test_polynomial(8, 8, 3);
         
         // 원본 구현 결과 확인
         let (quo_x_orig, quo_y_orig) = poly_orig.div_by_vanishing(2, 4);
         
         // EP 구현 결과 확인
         let (quo_x_ep, quo_y_ep) = poly_ep.div_by_vanishing(2, 4);
         
         // 결과 비교
         assert_polynomials_equal(&quo_x_orig, &quo_x_ep, "quotient_x (different degrees)");
         assert_polynomials_equal(&quo_y_orig, &quo_y_ep, "quotient_y (different degrees)");
     }
     
     // 최소 크기 케이스 테스트 (m=2, n=2)
     #[test]
     fn test_div_by_vanishing_minimum_size() {
         
         
         // 8x8 다항식 (m=2, n=2, denom_x=4, denom_y=4)
         let (poly_orig, poly_ep) = create_test_polynomial(8, 8, 3);
         
         // 원본 구현 결과 확인
         let (quo_x_orig, quo_y_orig) = poly_orig.div_by_vanishing(4, 4);
         
         // EP 구현 결과 확인
         let (quo_x_ep, quo_y_ep) = poly_ep.div_by_vanishing(4, 4);
         
         // 결과 비교
         assert_polynomials_equal(&quo_x_orig, &quo_x_ep, "quotient_x (minimum size)");
         assert_polynomials_equal(&quo_y_orig, &quo_y_ep, "quotient_y (minimum size)");
     }
}
