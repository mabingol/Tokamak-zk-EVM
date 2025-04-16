use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_core::traits::{Arithmetic, FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_bls12_381::symbol::bls12_381::FieldSymbol;
use icicle_core::ntt;
use icicle_core::symbol::Symbol;
use icicle_core::program::{Instruction, PreDefinedProgram, Program, ReturningValueProgram};
use icicle_bls12_381::program::bls12_381::FieldProgram;
use icicle_core::vec_ops::{VecOps, VecOpsConfig, execute_program, transpose_matrix};
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice, DeviceSlice, DeviceVec};
use std::{
    cmp,
    ops::{Add, AddAssign, Mul, Sub, Neg},
};
use rayon::prelude::*;

use crate::polynomials::{BivariatePolynomial, DensePolynomialExt};

fn _find_size_as_twopower(target_x_size: usize, target_y_size: usize) -> (usize, usize) {
    // Problem: find min{m: x_size*2^m >= target_x_size} and min{n: y_size*2^n >= target_y_size}
    if target_x_size == 0 || target_y_size == 0 {
        panic!("Invalid target sizes for resize")
    }
    let mut new_x_size = target_x_size;
    let mut new_y_size = target_y_size;
    if target_x_size.is_power_of_two() == false {
        new_x_size = 1 << (usize::BITS - target_x_size.leading_zeros());
    }
    if target_y_size.is_power_of_two() == false {
        new_y_size = 1 << (usize::BITS - target_y_size.leading_zeros());
    }
    (new_x_size, new_y_size)
}

// Method to divide a univariate polynomial by (X-x)
fn _div_uni_coeffs_by_ruffini(poly_coeffs_vec: &[ScalarField], x: ScalarField) -> (Vec<ScalarField>, ScalarField) {
    let len = poly_coeffs_vec.len();
    let mut q_coeffs_vec = vec![ScalarField::zero(); len];
    let mut b = poly_coeffs_vec[len - 1];
    q_coeffs_vec[len - 2] = b;
    for i in 3.. len + 1 {
        b = poly_coeffs_vec[len - i + 1] + b*x;
        q_coeffs_vec[len - i] = b;
    }
    let r = poly_coeffs_vec[0] + b*x;
    (q_coeffs_vec, r)
}


pub struct DensePolynomialExtEP {
    pub poly: DensePolynomial,
    pub x_degree: i64,
    pub y_degree: i64,
    pub x_size: usize,
    pub y_size: usize,
}

impl DensePolynomialExtEP {
    // Inherit DensePolynomial
    pub fn print(&self) {
        unsafe {
            self.poly.print()
        }
    }
    // Inherit DensePolynomial
    pub fn coeffs_mut_slice(&mut self) -> &mut DeviceSlice<ScalarField> {
        unsafe {
            self.poly.coeffs_mut_slice()          
        }
    }

    // Method to get the degree of the polynomial.
    pub fn degree(&self) -> (i64, i64) {
        (self.x_degree, self.y_degree)
    }
}

// impl Drop for DensePolynomialExtEP {
//     fn drop(&mut self) {
//         unsafe {
//             delete(self.poly);
//             delete(self.x_degree);
//             delete(self.y_degree);
//         }
//     }
// }

impl Clone for DensePolynomialExtEP {
    fn clone(&self) -> Self {
        Self {
            poly: self.poly.clone(),
            x_degree: self.x_degree.clone(),
            y_degree: self.y_degree.clone(),
            x_size: self.x_size.clone(),
            y_size: self.y_size.clone(),
        }
    }
}

impl Add for &DensePolynomialExtEP {
    type Output = DensePolynomialExtEP;
    fn add(self: Self, rhs: Self) -> Self::Output {
        let mut lhs_ext = self.clone();
        let mut rhs_ext = rhs.clone();
        if self.x_size != rhs.x_size || self.y_size != rhs.y_size {
            let target_x_size = cmp::max(self.x_size, rhs.x_size);
            let target_y_size = cmp::max(self.y_size, rhs.y_size);
            lhs_ext.resize(target_x_size, target_y_size);
            rhs_ext.resize(target_x_size, target_y_size);
        }
        let out_poly = &lhs_ext.poly + &rhs_ext.poly;
        let x_size = lhs_ext.x_size;
        let y_size = lhs_ext.y_size;
        //let (x_degree, y_degree) = DensePolynomialExtEP::find_degree(&out_poly, x_size, y_size);
        DensePolynomialExtEP {
            poly: out_poly,
            x_degree: x_size as i64 - 1,
            y_degree: y_size as i64 - 1,
            x_size,
            y_size,
        }
    }
}

impl AddAssign<&DensePolynomialExtEP> for DensePolynomialExtEP {
    fn add_assign(&mut self, rhs: &DensePolynomialExtEP) {
        let mut lhs_ext = self.clone();
        let mut rhs_ext = rhs.clone();
        if self.x_size != rhs.x_size || self.y_size != rhs.y_size {
            let target_x_size = cmp::max(self.x_size, rhs.x_size);
            let target_y_size = cmp::max(self.y_size, rhs.y_size);
            lhs_ext.resize(target_x_size, target_y_size);
            rhs_ext.resize(target_x_size, target_y_size);
        }
        self.poly = &lhs_ext.poly + &rhs_ext.poly;
        self.x_size = lhs_ext.x_size;
        self.y_size = lhs_ext.y_size;
        //let (x_degree, y_degree) = DensePolynomialExtEP::find_degree(&self.poly, self.x_size, self.y_size);
        self.x_degree = self.x_size as i64 - 1;
        self.y_degree = self.y_size as i64 - 1;
    }
}

impl Sub for &DensePolynomialExtEP {
    type Output = DensePolynomialExtEP;

    fn sub(self: Self, rhs: Self) -> Self::Output {
        let mut lhs_ext = self.clone();
        let mut rhs_ext = rhs.clone();
        if self.x_size != rhs.x_size || self.y_size != rhs.y_size {
            let target_x_size = cmp::max(self.x_size, rhs.x_size);
            let target_y_size = cmp::max(self.y_size, rhs.y_size);
            lhs_ext.resize(target_x_size, target_y_size);
            rhs_ext.resize(target_x_size, target_y_size);
        }
        let out_poly = &lhs_ext.poly - &rhs_ext.poly;
        let x_size = lhs_ext.x_size;
        let y_size = lhs_ext.y_size;
        //let (x_degree, y_degree) = DensePolynomialExtEP::find_degree(&out_poly, x_size, y_size);
        DensePolynomialExtEP {
            poly: out_poly,
            x_degree: x_size as i64 - 1,
            y_degree: y_size as i64 - 1,
            x_size,
            y_size,
        }
    }
}

// impl Mul for &DensePolynomialExtEP {
//     type Output = DensePolynomialExtEP;

//     fn mul(self: Self, rhs: Self) -> Self::Output {
//         self._mul(rhs)
//     }
// }

// poly * scalar
impl Mul<&ScalarField> for &DensePolynomialExtEP {
    type Output = DensePolynomialExtEP;

    fn mul(self: Self, rhs: &ScalarField) -> Self::Output {
        DensePolynomialExtEP {
            poly: &self.poly * rhs,
            x_degree: self.x_degree,
            y_degree: self.y_degree,
            x_size: self.x_size,
            y_size: self.y_size,
        }
    }
}

// scalar * poly
impl Mul<&DensePolynomialExtEP> for &ScalarField {
    type Output = DensePolynomialExtEP;

    fn mul(self: Self, rhs: &DensePolynomialExtEP) -> Self::Output {
        DensePolynomialExtEP {
            poly: self * &rhs.poly,
            x_degree: rhs.x_degree,
            y_degree: rhs.y_degree,
            x_size: rhs.x_size,
            y_size: rhs.y_size,
        }
    }
}

// impl Neg for &DensePolynomialExtEP {
//     type Output = DensePolynomialExtEP;

//     fn neg(self: Self) -> Self::Output {
//         self._neg()
//     }
// }


pub trait BivariatePolynomialEP
where
    Self::Field: FieldImpl,
    Self::FieldConfig: FieldConfig,
{
    type Field: FieldImpl;
    type FieldConfig: FieldConfig;

    // Methods to create polynomials from coefficients or roots-of-unity evaluations.
    fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, x_size: usize, y_size: usize) -> Self;
    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, x_size: usize, y_size: usize, coset_x: Option<&Self::Field>, coset_y: Option<&Self::Field>) -> Self;
    // Method to evaluate the polynomial over the roots-of-unity domain for power-of-two sized domain
    fn to_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, coset_x: Option<&Self::Field>, coset_y: Option<&Self::Field>, evals: &mut S);
    
    fn find_degree(coeffs: &DensePolynomial, x_size: usize, y_size: usize) -> (i64, i64);

    // Method to divide this polynomial by vanishing polynomials 'X^{x_degree}-1' and 'Y^{y_degree}-1'.
    fn div_by_vanishing(&self, x_degree: i64, y_degree: i64) -> (Self, Self) where Self: Sized;

    // Method to divide this polynomial by (X-x) and (Y-y)
    fn div_by_ruffini(&self, x: Self::Field, y: Self::Field) -> (Self, Self, Self::Field) where Self: Sized;

    // // Methods to add or subtract a monomial in-place.
    // fn add_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);
    // fn sub_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);

    // Method to shift coefficient indicies. The same effect as multiplying a monomial X^iY^j.
    fn mul_monomial(&self, x_exponent: usize, y_exponent: usize) -> Self;

    fn resize(&mut self, target_x_size: usize, target_y_size: usize);
    fn optimize_size(&mut self);

    // Method to slice the polynomial, creating a sub-polynomial.
    fn _slice_coeffs_into_blocks(&self, num_blocks_x: usize, num_blocks_y: usize, blocks_raw: &mut Vec<Vec<Self::Field>> );

    // // Methods to return new polynomials containing only the even or odd terms.
    // fn even_x(&self) -> Self;
    // fn even_y(&self) -> Self;
    // fn odd_y(&self) -> Self;
    // fn odd_y(&self) -> Self;

    // Method to evaluate the polynomial at a given domain point.
    fn eval_x(&self, x: &Self::Field) -> Self;

    // Method to evaluate the polynomial at a given domain point.
    fn eval_y(&self, y: &Self::Field) -> Self;

    fn eval(&self, x: &Self::Field, y: &Self::Field) -> Self::Field;

    // // Method to evaluate the polynomial over a domain and store the results.
    // fn eval_on_domain<D_x: HostOrDeviceSlice<Self::Field> + ?Sized, D_y: HostOrDeviceSlice<Self::Field> + ?Sized, E: HostOrDeviceSlice<Self::Field> + ?Sized>(
    //     &self,
    //     domain_x: &D_x,
    //     domain_y: &D_y,
    //     evals: &mut E,
    // );

    // Method to retrieve a coefficient at a specific index.
    fn get_coeff(&self, idx_x: u64, idx_y: u64) -> Self::Field;
    // fn get_nof_coeffs_x(&self) -> u64;
    // fn get_nof_coeffs_y(&self) -> u64;
    
    // Method to retrieve a univariate polynomial of x as the coefficient of the 'idx_y'-th power of y.
    fn get_univariate_polynomial_x(&self, idx_y:u64) -> Self;
    // Method to retrieve a univariate polynomial of y as the coefficient of the 'idx_x'-th power of x.
    fn get_univariate_polynomial_y(&self, idx_x:u64) -> Self;

    // Method to copy coefficients into a provided slice.
    fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, start_idx: u64, coeffs: &mut S);

    fn _mul(&self, rhs: &Self) -> Self;
    // Method to divide this polynomial by another, returning quotient and remainder.
    fn divide_x(&self, denominator: &Self) -> (Self, Self) where Self: Sized;

    // Method to divide this polynomial by another, returning quotient and remainder.
    fn divide_y(&self, denominator: &Self) -> (Self, Self) where Self: Sized;

    fn _neg(&self) -> Self;

}

impl BivariatePolynomialEP for DensePolynomialExtEP {
    type Field = ScalarField;
    type FieldConfig = ScalarCfg;

    fn find_degree(poly: &DensePolynomial, x_size: usize, y_size: usize) -> (i64, i64) {
        // Determine which dimension is smaller
        let (min_size, is_min_x) = if x_size <= y_size {
            (x_size, true)
        } else {
            (y_size, false)
        };
    
        let mut max_x_degree: i64 = -1;
        let mut max_y_degree: i64 = -1;
        let vec_ops_cfg = VecOpsConfig::default();
    
        // Create a program that will process polynomial data
        // This is a simplified implementation since we can't fully replicate polynomial degree calculation
        let process_poly_program = FieldProgram::new(
            |vars: &mut Vec<FieldSymbol>| {
                // Program logic would go here
                // Since FieldSymbol doesn't support direct comparison, we'll use a different approach
                
                // We're creating a simple pass-through program
                // In a real implementation, this would contain logic to analyze polynomial data
                vars[1] = vars[0]; // Simply copy input to output as placeholder
            },
            2 // Input and output parameters
        ).unwrap();
    
        for ind in 0..min_size as i64 {
            let sub_poly = if is_min_x {
                // sub-polynomial of y
                poly.slice(ind as u64, x_size as u64, y_size as u64)
            } else {
                // sub-polynomial of x
                poly.slice(ind as u64 * x_size as u64, 1, x_size as u64)
            };
            
            // Calculate degree using the original method
            let curr_degree = sub_poly.degree() as i64;
            
            // Create input for our program - a single value representing if polynomial is non-zero
            let has_nonzero = if curr_degree > -1 { 
                ScalarField::one() 
            } else { 
                ScalarField::zero() 
            };
            
            // Prepare for result
            let mut result_value = ScalarField::zero();
            
            // Prepare slices for execute_program
            let input_vec = vec![has_nonzero];
            let input_slice = HostSlice::from_slice(&input_vec);
            let mut output_array = [result_value];
            let output_slice = HostSlice::from_mut_slice(&mut output_array);
            
            // Execute the program
            let mut parameters = vec![input_slice, output_slice];
            execute_program(&mut parameters, &process_poly_program, &vec_ops_cfg).unwrap();
            
            // Process the result 
            // We're still using curr_degree since our program is just a placeholder
            if curr_degree > -1 {
                if is_min_x {
                    max_y_degree = std::cmp::max(curr_degree, max_y_degree);
                    max_x_degree = std::cmp::max(ind, max_x_degree);
                } else {
                    max_x_degree = std::cmp::max(curr_degree, max_x_degree);
                    max_y_degree = std::cmp::max(ind, max_y_degree);
                }
            }
        }
        
        (std::cmp::max(max_x_degree, 0), std::cmp::max(max_y_degree, 0))
    }

    fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, x_size: usize, y_size: usize) -> Self {
        if x_size == 0 || y_size == 0 {
            panic!("Invalid matrix size for from_coeffs");
        }
        if x_size.is_power_of_two() == false || y_size.is_power_of_two() == false {
            println!("x_size, y_size: {:?}, {:?}", x_size, y_size);
            panic!("The input sizes for from_coeffs must be powers of two.")
        }
        let poly = DensePolynomial::from_coeffs(coeffs, x_size as usize * y_size as usize);
        //let (x_degree, y_degree) = DensePolynomialExtEP::_find_degree(&poly, x_size, y_size);
        Self {
            poly,
            x_degree: x_size as i64 - 1,
            y_degree: y_size as i64 - 1,
            x_size,
            y_size,
        }
    }

    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(
        evals: &S, 
        x_size: usize, 
        y_size: usize, 
        coset_x: Option<&Self::Field>, 
        coset_y: Option<&Self::Field>
    ) -> Self {
        if x_size == 0 || y_size == 0 {
            panic!("Invalid matrix size for from_rou_evals");
        }
        if x_size.is_power_of_two() == false || y_size.is_power_of_two() == false {
            panic!("The input sizes for from_rou_evals must be powers of two.")
        }
    
        let size = x_size * y_size;
    
        ntt::initialize_domain::<Self::Field>(
            ntt::get_root_of_unity::<Self::Field>(
                size.try_into().unwrap(),
            ),
            &ntt::NTTInitDomainConfig::default(),
        ).unwrap();
    
        let mut coeffs = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
        let mut cfg = ntt::NTTConfig::<Self::Field>::default();
        
        // IFFT along X
        cfg.batch_size = y_size as i32;
        cfg.columns_batch = false;
        ntt::ntt(evals, ntt::NTTDir::kInverse, &cfg, &mut coeffs).unwrap();
        
        // IFFT along Y
        cfg.batch_size = x_size as i32;
        cfg.columns_batch = true;
        ntt::ntt_inplace(&mut coeffs, ntt::NTTDir::kInverse, &cfg).unwrap();
    
        let mut scaled_coeffs = coeffs;
        let vec_ops_cfg = VecOpsConfig::default();
    
        let mul_program = FieldProgram::new(
            |vars: &mut Vec<FieldSymbol>| {
                vars[2] = vars[0] * vars[1];
            },
            3 
        ).unwrap();
    
        if let Some(_factor) = coset_x {
            let factor = _factor.inv();
            let mut _right_scale = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            let mut scaler = Self::Field::one();
            
            for ind in 0..x_size {
                _right_scale[ind * y_size .. (ind+1) * y_size].copy_from_host(HostSlice::from_slice(&vec![scaler; y_size])).unwrap();
                scaler = scaler.mul(factor);
            }
            
            let mut right_scale = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            Self::FieldConfig::transpose(&_right_scale, x_size as u32, y_size as u32, &mut right_scale, &vec_ops_cfg).unwrap();
            
            let mut temp = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            
            let mut host_a = vec![Self::Field::zero(); size];
            let mut host_b = vec![Self::Field::zero(); size];
            let mut host_result = vec![Self::Field::zero(); size];
            
            scaled_coeffs.copy_to_host(HostSlice::from_mut_slice(&mut host_a)).unwrap();
            right_scale.copy_to_host(HostSlice::from_mut_slice(&mut host_b)).unwrap();
            
            let mut parameters = vec![
                HostSlice::from_slice(&host_a),
                HostSlice::from_slice(&host_b),
                HostSlice::from_mut_slice(&mut host_result)
            ];
            
            execute_program(&mut parameters, &mul_program, &vec_ops_cfg).unwrap();
            
            temp.copy_from_host(HostSlice::from_slice(&host_result)).unwrap();
            scaled_coeffs = temp;
        }
    
        if let Some(_factor) = coset_y {
            let factor = _factor.inv();
            let mut left_scale = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            let mut scaler = Self::Field::one();
            
            for ind in 0..y_size {
                left_scale[ind * x_size .. (ind+1) * x_size].copy_from_host(HostSlice::from_slice(&vec![scaler; x_size])).unwrap();
                scaler = scaler.mul(factor);
            }
            
            let mut temp = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            
            let mut host_a = vec![Self::Field::zero(); size];
            let mut host_b = vec![Self::Field::zero(); size];
            let mut host_result = vec![Self::Field::zero(); size];
            
            scaled_coeffs.copy_to_host(HostSlice::from_mut_slice(&mut host_a)).unwrap();
            left_scale.copy_to_host(HostSlice::from_mut_slice(&mut host_b)).unwrap();
            
            let mut parameters = vec![
                HostSlice::from_slice(&host_a),
                HostSlice::from_slice(&host_b),
                HostSlice::from_mut_slice(&mut host_result)
            ];
            
            execute_program(&mut parameters, &mul_program, &vec_ops_cfg).unwrap();
            
            temp.copy_from_host(HostSlice::from_slice(&host_result)).unwrap();
            scaled_coeffs = temp;
        }
    
        DensePolynomialExtEP::from_coeffs(&scaled_coeffs, x_size, y_size)
    }

    fn to_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(
        &self,
        coset_x: Option<&Self::Field>,
        coset_y: Option<&Self::Field>,
        evals: &mut S,
    ) {
        let size = self.x_size * self.y_size;
        if evals.len() < size {
            panic!("Insufficient buffer length for to_rou_evals")
        }
        
        // Allocate memory for coefficients
        let mut coeffs = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
        self.copy_coeffs(0, &mut coeffs);
        
        // 임시 호스트 버퍼를 사용하여 복사
        let mut host_buffer = vec![Self::Field::zero(); size];
        coeffs.copy_to_host(HostSlice::from_mut_slice(&mut host_buffer)).unwrap();
        
        // 이제 host_buffer에서 scaled_coeffs로 복사
        let mut scaled_coeffs = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
        scaled_coeffs.copy_from_host(HostSlice::from_slice(&host_buffer)).unwrap();
        
        let vec_ops_cfg = VecOpsConfig::default();
        
        // Create a multiplication program to be reused
        let mul_program = FieldProgram::new(
            |vars: &mut Vec<FieldSymbol>| {
                vars[2] = vars[0] * vars[1];
            },
            3 
        ).unwrap();
        
        // Handle coset_x scaling
        if let Some(factor) = coset_x {
            let mut left_scale = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            let mut scaler = Self::Field::one();
            
            for ind in 0..self.x_size {
                left_scale[ind * self.y_size .. (ind+1) * self.y_size].copy_from_host(HostSlice::from_slice(&vec![scaler; self.y_size])).unwrap();
                scaler = scaler.mul(*factor);
            }
            
            // Use the multiplication program with host slices
            let mut host_a = vec![Self::Field::zero(); size];
            let mut host_b = vec![Self::Field::zero(); size];
            let mut host_result = vec![Self::Field::zero(); size];
            
            scaled_coeffs.copy_to_host(HostSlice::from_mut_slice(&mut host_a)).unwrap();
            left_scale.copy_to_host(HostSlice::from_mut_slice(&mut host_b)).unwrap();
            
            let mut parameters = vec![
                HostSlice::from_slice(&host_a),
                HostSlice::from_slice(&host_b),
                HostSlice::from_mut_slice(&mut host_result)
            ];
            
            execute_program(&mut parameters, &mul_program, &vec_ops_cfg).unwrap();
            
            let mut temp = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            temp.copy_from_host(HostSlice::from_slice(&host_result)).unwrap();
            scaled_coeffs = temp;
        }
        
        // Handle coset_y scaling
        if let Some(factor) = coset_y {
            let mut _right_scale = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            let mut scaler = Self::Field::one();
            
            for ind in 0..self.y_size {
                _right_scale[ind * self.x_size .. (ind+1) * self.x_size].copy_from_host(HostSlice::from_slice(&vec![scaler; self.x_size])).unwrap();
                scaler = scaler.mul(*factor);
            }
            
            let mut right_scale = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            Self::FieldConfig::transpose(&_right_scale, self.y_size as u32, self.x_size as u32, &mut right_scale, &vec_ops_cfg).unwrap();
            
            // Use the multiplication program with host slices
            let mut host_a = vec![Self::Field::zero(); size];
            let mut host_b = vec![Self::Field::zero(); size];
            let mut host_result = vec![Self::Field::zero(); size];
            
            scaled_coeffs.copy_to_host(HostSlice::from_mut_slice(&mut host_a)).unwrap();
            right_scale.copy_to_host(HostSlice::from_mut_slice(&mut host_b)).unwrap();
            
            let mut parameters = vec![
                HostSlice::from_slice(&host_a),
                HostSlice::from_slice(&host_b),
                HostSlice::from_mut_slice(&mut host_result)
            ];
            
            execute_program(&mut parameters, &mul_program, &vec_ops_cfg).unwrap();
            
            let mut temp = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
            temp.copy_from_host(HostSlice::from_slice(&host_result)).unwrap();
            scaled_coeffs = temp;
        }
        
        // NTT operations
        ntt::initialize_domain::<Self::Field>(
            ntt::get_root_of_unity::<Self::Field>(
                size.try_into().unwrap(),
            ),
            &ntt::NTTInitDomainConfig::default(),
        ).unwrap();
        
        let mut cfg = ntt::NTTConfig::<Self::Field>::default();
        
        // FFT along X
        cfg.batch_size = self.y_size as i32;
        cfg.columns_batch = true;
        ntt::ntt(&scaled_coeffs, ntt::NTTDir::kForward, &cfg, evals).unwrap();
        
        // FFT along Y
        cfg.batch_size = self.x_size as i32;
        cfg.columns_batch = false;
        ntt::ntt_inplace(evals, ntt::NTTDir::kForward, &cfg).unwrap();
        
        ntt::release_domain::<Self::Field>().unwrap();
    }

    fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, start_idx: u64, coeffs: &mut S) {
        self.poly.copy_coeffs(start_idx, coeffs);
    }

    fn _neg(&self) -> Self {
        let zero_vec = vec![Self::Field::zero(); 1];
        let zero_poly = DensePolynomialExtEP::from_coeffs(HostSlice::from_slice(&zero_vec), 1, 1);
        &zero_poly - self
    }

    fn _slice_coeffs_into_blocks(&self, num_blocks_x: usize, num_blocks_y: usize, blocks: &mut Vec<Vec<Self::Field>> ) {

        if self.x_size % num_blocks_x != 0 || self.y_size % num_blocks_y != 0 {
            panic!("Matrix size must be exactly divisible by the number of blocks.");
        }
        if blocks.len() != num_blocks_x * num_blocks_y {
            panic!("Incorrect length of the vector to store the result.")
        }
        let block_x_size = self.x_size / num_blocks_x;
        let block_y_size = self.y_size / num_blocks_y;

        let mut orig_coeffs_vec = vec![Self::Field::zero(); self.x_size * self.y_size];
        let orig_coeffs = HostSlice::from_mut_slice(&mut orig_coeffs_vec);
        self.poly.copy_coeffs(0, orig_coeffs);

        for row_idx in 0..self.y_size{
            let row_vec = &orig_coeffs_vec[row_idx * self.x_size .. (row_idx + 1) * self.x_size];
            for col_idx in 0..self.x_size {
                let block_idx = (col_idx / block_x_size) + num_blocks_x * (row_idx / block_y_size);
                let in_block_idx = (col_idx % block_x_size) + block_x_size * (row_idx % block_y_size);
                blocks[block_idx][in_block_idx] = row_vec[col_idx].clone();
            }
        }

    }

    fn eval_x(&self, x: &Self::Field) -> Self {
        let mut result_slice = vec![Self::Field::zero(); self.y_size];
        let result = HostSlice::from_mut_slice(&mut result_slice);

        for offset in 0..self.y_degree as usize + 1 {
            let sub_xpoly = self.get_univariate_polynomial_x(offset as u64);
            result[offset] = sub_xpoly.poly.eval(x);
        }

        DensePolynomialExtEP::from_coeffs(result, 1, self.y_size)
    }

    fn eval_y(&self, y: &Self::Field) -> Self {
        let mut result_slice = vec![Self::Field::zero(); self.x_size];
        let result = HostSlice::from_mut_slice(&mut result_slice);

        for offset in 0..self.x_degree as usize + 1 {
            let sub_ypoly = self.get_univariate_polynomial_y(offset as u64); 
            result[offset] = sub_ypoly.poly.eval(y);
        }
        DensePolynomialExtEP::from_coeffs(result, self.x_size, 1)
    }

    fn eval(&self, x: &Self::Field, y: &Self::Field) -> Self::Field {
        let res1 = self.eval_x(x);
        let res2 = res1.eval_y(y);
        if !(res2.x_degree == 0 && res2.y_degree == 0) {
            panic!("The evaluation is not a constant.");
        } else {
            res2.get_coeff(0,0)
        }
    }

    fn get_coeff(&self, idx_x: u64, idx_y: u64) -> Self::Field {
        if !(idx_x <= self.x_size as u64 && idx_y <= self.y_size as u64){
            panic!("The index at which to get a coefficient exceeds the coefficient size.");
        }
        let idx = idx_x + idx_y * self.x_size as u64;
        self.poly.get_coeff(idx)
    }

    fn get_univariate_polynomial_x(&self, idx_y:u64) -> Self {
        Self {
            poly: self.poly.slice(idx_y * self.x_size as u64, 1, self.x_size as u64),
            x_size: self.x_size.clone(),
            y_size: 1,
            x_degree: self.x_degree.clone(),
            y_degree: 0,
        }
    }

    fn get_univariate_polynomial_y(&self, idx_x:u64) -> Self {
        Self {
            poly: self.poly.slice(idx_x, self.x_size as u64, self.y_size as u64),
            x_size: 1,
            y_size: self.y_size.clone(),
            x_degree: 0,
            y_degree: self.y_degree.clone(),
        }
    }

    
    fn resize(&mut self, target_x_size: usize, target_y_size: usize) {
        // Check and calculate sizes as power of 2
        let (new_x_size, new_y_size) = _find_size_as_twopower(target_x_size, target_y_size);
        
        // If sizes are already as requested, return early
        if self.x_size == new_x_size && self.y_size == new_y_size {
            return;
        }
        
        // Store current coefficients in host memory
        let orig_size = self.x_size * self.y_size;
        let mut orig_coeffs_vec = vec![ScalarField::zero(); orig_size];
        let orig_coeffs = HostSlice::from_mut_slice(&mut orig_coeffs_vec);
        self.copy_coeffs(0, orig_coeffs);
        
        // Allocate memory for the resized polynomial
        let new_size = new_x_size * new_y_size;
        let mut res_coeffs_vec = vec![ScalarField::zero(); new_size];
        
        // Since we can't use FieldSymbol.to_u32(), we'll approach this differently
        // We'll create a row-by-row copy program instead
        
        let copy_program = FieldProgram::new(
            |vars: &mut Vec<FieldSymbol>| {
                vars[1] = vars[0];
            },
            2 
        ).unwrap();
        
        let vec_ops_cfg = VecOpsConfig::default();
        
        // Copy row by row, similar to the original implementation
        for i in 0..std::cmp::min(self.y_size, new_y_size) {
            let each_x_size = std::cmp::min(self.x_size, new_x_size);
            
            let src_start = self.x_size * i;
            let dst_start = new_x_size * i;
            
            // Create slices for this row
            let src_row = &orig_coeffs_vec[src_start..src_start + each_x_size];
            let mut dst_row = vec![ScalarField::zero(); each_x_size]; 
            
            // Set up parameters for this row
            let mut parameters = vec![
                HostSlice::from_slice(src_row),
                HostSlice::from_mut_slice(&mut dst_row)
            ];
            
            // Execute copy program for this row
            execute_program(&mut parameters, &copy_program, &vec_ops_cfg).unwrap();
            
            // Copy the results back to the result vector
            res_coeffs_vec[dst_start..dst_start + each_x_size].copy_from_slice(&dst_row);
        }
        
        // Update polynomial with new coefficients and dimensions
        let res_coeffs = HostSlice::from_slice(&res_coeffs_vec);
        self.poly = DensePolynomial::from_coeffs(res_coeffs, new_size);
        self.x_size = new_x_size;
        self.y_size = new_y_size;
    }

    fn optimize_size(&mut self) {
        let target_x_size = self.x_degree as usize + 1;
        let target_y_size = self.y_degree as usize + 1;
        self.resize(target_x_size, target_y_size);
    }

    fn mul_monomial(&self, x_exponent: usize, y_exponent: usize) -> Self {
       if x_exponent == 0 && y_exponent == 0 {
            self.clone()
        } else {
            let mut orig_coeffs_vec = Vec::<Self::Field>::with_capacity(self.x_size * self.y_size);
            unsafe{orig_coeffs_vec.set_len(self.x_size * self.y_size);}
            let orig_coeffs = HostSlice::from_mut_slice(&mut orig_coeffs_vec);
            self.copy_coeffs(0, orig_coeffs);

            let target_x_size = self.x_degree as usize + x_exponent + 1;
            let target_y_size = self.y_degree as usize + y_exponent + 1;
            let (new_x_size, new_y_size) = _find_size_as_twopower(target_x_size, target_y_size);
            let new_size: usize = new_x_size * new_y_size;
            
            let mut res_coeffs_vec = vec![Self::Field::zero(); new_size];
            for i in 0 .. self.y_size {
                res_coeffs_vec[new_x_size * (i + y_exponent) + x_exponent .. new_x_size * (i + y_exponent) + self.x_size + x_exponent].copy_from_slice(
                    &orig_coeffs_vec[self.x_size * i .. self.x_size * (i+1)]
                );
            }

            let res_coeffs = HostSlice::from_slice(&res_coeffs_vec);
            
            DensePolynomialExtEP::from_coeffs(res_coeffs, new_x_size, new_y_size)
        }
    }

    fn _mul(&self, rhs: &Self) -> Self {
        // 특수 케이스 처리
        let (lhs_x_degree, lhs_y_degree) = self.degree();
        let (rhs_x_degree, rhs_y_degree) = rhs.degree();
        
        // 상수 다항식 처리
        if lhs_x_degree + lhs_y_degree == 0 && rhs_x_degree + rhs_y_degree > 0 {
            return &(rhs.clone()) * &(self.get_coeff(0, 0));
        }
        if rhs_x_degree + rhs_y_degree == 0 && lhs_x_degree + lhs_y_degree > 0 {
            return &(self.clone()) * &(rhs.get_coeff(0,0));
        }
        if rhs_x_degree + rhs_y_degree == 0 && lhs_x_degree + lhs_y_degree == 0 {
            let out_coeffs_vec = vec![self.get_coeff(0,0) * rhs.get_coeff(0,0); 1];
            let out_coeffs = HostSlice::from_slice(&out_coeffs_vec);
            return DensePolynomialExtEP::from_coeffs(out_coeffs, 1, 1);
        }
        
        // 곱셈 후 예상 차수 계산
        let x_degree = lhs_x_degree + rhs_x_degree;
        let y_degree = lhs_y_degree + rhs_y_degree;
        let target_x_size = x_degree as usize + 1;
        let target_y_size = y_degree as usize + 1;
        
        // 크기 조정
        let mut lhs_ext = self.clone();
        let mut rhs_ext = rhs.clone();
        lhs_ext.resize(target_x_size, target_y_size);
        rhs_ext.resize(target_x_size, target_y_size);
        
        let x_size = lhs_ext.x_size;
        let y_size = lhs_ext.y_size;
        let extended_size = x_size * y_size;
        
        // execute_program을 사용한 FFT 기반 곱셈 구현
        
        // 1. 계수에서 평가값으로 변환 (FFT)
        let mut lhs_evals = DeviceVec::<ScalarField>::device_malloc(extended_size).unwrap();
        let mut rhs_evals = DeviceVec::<ScalarField>::device_malloc(extended_size).unwrap();
        
        // FFT 계산 (기존 코드 재사용)
        lhs_ext.to_rou_evals(None, None, &mut lhs_evals);
        rhs_ext.to_rou_evals(None, None, &mut rhs_evals);
        
        // 2. 배치 단위로 원소별 곱셈 수행
        // pointwise 곱셈을 위한 FieldProgram 정의
        let pointwise_mul_program = FieldProgram::new(
            |symbols: &mut Vec<FieldSymbol>| {
                // 원소별 곱셈 수행
                symbols[2] = symbols[0] * symbols[1];
            },
            3 // 입력 A, 입력 B, 출력 C
        ).unwrap();
        
        let vec_ops_cfg = VecOpsConfig::default();
        
        // 2-a. HostSlice로 lhs_evals와 rhs_evals 가져오기
        let mut lhs_evals_host = vec![ScalarField::zero(); extended_size];
        let mut rhs_evals_host = vec![ScalarField::zero(); extended_size];
        lhs_evals.copy_to_host(HostSlice::from_mut_slice(&mut lhs_evals_host)).unwrap();
        rhs_evals.copy_to_host(HostSlice::from_mut_slice(&mut rhs_evals_host)).unwrap();
        
        // 2-b. 결과 저장 공간 할당
        let mut result_evals_host = vec![ScalarField::zero(); extended_size];
        
        // 2-c. 배치 크기 정의 (최적의 배치 크기는 하드웨어에 따라 다름)
        let batch_size = 1024; // 적절한 배치 크기 선택
        
        // 2-d. 배치 단위로 pointwise 곱셈 수행
        for start_idx in (0..extended_size).step_by(batch_size) {
            let end_idx = std::cmp::min(start_idx + batch_size, extended_size);
            let current_batch_size = end_idx - start_idx;
            
            // 현재 배치의 입력 슬라이스
            let lhs_batch = &lhs_evals_host[start_idx..end_idx];
            let rhs_batch = &rhs_evals_host[start_idx..end_idx];
            
            // 현재 배치의 출력 슬라이스
            let result_batch = &mut result_evals_host[start_idx..end_idx];
            
            // 배치에 대한 pointwise 곱셈 수행
            let mut parameters = vec![
                HostSlice::from_slice(lhs_batch),
                HostSlice::from_slice(rhs_batch),
                HostSlice::from_mut_slice(result_batch)
            ];
            
            execute_program(&mut parameters, &pointwise_mul_program, &vec_ops_cfg).unwrap();
        }
        
        // 3. 결과 DeviceVec 생성 및 데이터 복사
        let mut out_evals = DeviceVec::<ScalarField>::device_malloc(extended_size).unwrap();
        out_evals.copy_from_host(HostSlice::from_slice(&result_evals_host)).unwrap();
        
        // 4. IFFT를 사용하여 결과 계수 계산
        DensePolynomialExtEP::from_rou_evals(&out_evals, x_size, y_size, None, None)
    }

    fn divide_x(&self, denominator: &Self) -> (Self, Self) where Self: Sized {
        let (numer_x_degree, numer_y_degree) = self.degree();
        let (denom_x_degree, denom_y_degree) = denominator.degree();
        if denom_y_degree != 0 {
            panic!("Denominator for divide_x must be X-univariate");
        }
        if numer_x_degree < denom_x_degree{
            panic!("Numer.degree < Denom.degree for divide_x");
        }
        if denom_x_degree == 0 {
            if Self::Field::eq(&(denominator.get_coeff(0, 0).inv()), &Self::Field::zero()) {
                panic!("Divide by zero")
            }
            let rem_coeffs_vec = vec![Self::Field::zero(); 1];
            let rem_coeffs = HostSlice::from_slice(&rem_coeffs_vec);
            return (
                &(self.clone()) * &(denominator.get_coeff(0, 0).inv()),
                DensePolynomialExtEP::from_coeffs(rem_coeffs, 1, 1),
            );
        }

        let quo_x_degree = numer_x_degree - denom_x_degree;
        let quo_y_degree = numer_y_degree;
        let rem_x_degree = denom_x_degree - 1;
        let rem_y_degree = numer_y_degree;

        let quo_x_size = quo_x_degree as usize + 1;
        let quo_y_size = quo_y_degree as usize + 1;
        let rem_x_size = rem_x_degree as usize + 1;
        let rem_y_size = rem_y_degree as usize + 1;
        // let quo_x_size = next_power_of_two(quo_x_degree as usize + 1);
        // let quo_y_size = next_power_of_two(quo_y_degree as usize + 1);
        // let rem_x_size = next_power_of_two(rem_x_degree as usize + 1);
        // let rem_y_size = next_power_of_two(rem_y_degree as usize + 1);
        
        let quo_size = quo_x_size * quo_y_size;
        let rem_size = rem_x_size * rem_y_size;

        let mut quo_coeffs_vec = vec![Self::Field::zero(); quo_size];
        let mut rem_coeffs_vec = vec![Self::Field::zero(); rem_size];

        for offset in 0..self.y_degree as usize + 1 {
            let sub_xpoly = self.get_univariate_polynomial_x(offset as u64);
            let (sub_quo_poly, sub_rem_poly) = sub_xpoly.poly.divide(&denominator.poly);
            let mut sub_quo_coeffs_vec = vec![Self::Field::zero(); quo_x_size];
            let mut sub_rem_coeffs_vec = vec![Self::Field::zero(); rem_x_size];
            let sub_quo_coeffs = HostSlice::from_mut_slice(&mut sub_quo_coeffs_vec);
            let sub_rem_coeffs = HostSlice::from_mut_slice(&mut sub_rem_coeffs_vec);
            sub_quo_poly.copy_coeffs(0, sub_quo_coeffs);
            sub_rem_poly.copy_coeffs(0, sub_rem_coeffs);
            if offset <= quo_y_size {
                quo_coeffs_vec[offset * quo_x_size .. (offset + 1) * quo_x_size].copy_from_slice(&sub_quo_coeffs_vec);
            }
            if offset <= rem_y_size {
                rem_coeffs_vec[offset * rem_x_size .. (offset + 1) * rem_x_size].copy_from_slice(&sub_rem_coeffs_vec);
            }
        }

        let quo_coeffs = HostSlice::from_mut_slice(&mut quo_coeffs_vec);
        let rem_coeffs = HostSlice::from_mut_slice(&mut rem_coeffs_vec);
        (DensePolynomialExtEP::from_coeffs(quo_coeffs, quo_x_size, quo_y_size), DensePolynomialExtEP::from_coeffs(rem_coeffs, rem_x_size, rem_y_size))
    }


    fn divide_y(&self, denominator: &Self) -> (Self, Self) where Self: Sized {
        let (numer_x_degree, numer_y_degree) = self.degree();
        let (denom_x_degree, denom_y_degree) = denominator.degree();
        if denom_x_degree != 0 {
            panic!("Denominator for divide_y must be Y-univariate");
        }
        if numer_y_degree < denom_y_degree{
            panic!("Numer.y_degree < Denom.y_degree for divide_y");
        }
        if denom_y_degree == 0 {
            if Self::Field::eq(&(denominator.get_coeff(0, 0).inv()), &Self::Field::zero()) {
                panic!("Divide by zero")
            }
            let rem_coeffs_vec = vec![Self::Field::zero(); 1];
            let rem_coeffs = HostSlice::from_slice(&rem_coeffs_vec);
            return (
                &(self.clone()) * &(denominator.get_coeff(0, 0).inv()),
                DensePolynomialExtEP::from_coeffs(rem_coeffs, 1, 1),
            );
        }

        let quo_x_degree = numer_x_degree;
        let quo_y_degree = numer_y_degree - denom_y_degree;
        let rem_x_degree = numer_x_degree;
        let rem_y_degree = denom_y_degree - 1;
        let quo_x_size = quo_x_degree as usize + 1;
        let quo_y_size = quo_y_degree as usize + 1;
        let rem_x_size = rem_x_degree as usize + 1;
        let rem_y_size = rem_y_degree as usize + 1;
        let quo_size = quo_x_size * quo_y_size;
        let rem_size = rem_x_size * rem_y_size;

        let mut quo_coeffs_vec = vec![Self::Field::zero(); quo_size];
        let mut rem_coeffs_vec = vec![Self::Field::zero(); rem_size];

        for offset in 0..self.x_degree as usize + 1 {
            let sub_ypoly = self.get_univariate_polynomial_y(offset as u64);
            let (sub_quo_poly, sub_rem_poly) = sub_ypoly.poly.divide(&denominator.poly);
            let mut sub_quo_coeffs_vec = vec![Self::Field::zero(); quo_y_size];
            let mut sub_rem_coeffs_vec = vec![Self::Field::zero(); rem_y_size];
            let sub_quo_coeffs = HostSlice::from_mut_slice(&mut sub_quo_coeffs_vec);
            let sub_rem_coeffs = HostSlice::from_mut_slice(&mut sub_rem_coeffs_vec);
            sub_quo_poly.copy_coeffs(0, sub_quo_coeffs);
            sub_rem_poly.copy_coeffs(0, sub_rem_coeffs);
            if offset <= quo_x_size {
                quo_coeffs_vec[offset * quo_y_size .. (offset + 1) * quo_y_size].copy_from_slice(&sub_quo_coeffs_vec);
            }
            if offset <= rem_x_size {
                rem_coeffs_vec[offset * rem_y_size .. (offset + 1) * rem_y_size].copy_from_slice(&sub_rem_coeffs_vec);
            }
        }
        let quo_coeffs_tr = HostSlice::from_slice(&quo_coeffs_vec);
        let rem_coeffs_tr = HostSlice::from_slice(&rem_coeffs_vec);

        let mut quo_coeffs_vec2 = vec![Self::Field::zero(); quo_size];
        let quo_coeffs = HostSlice::from_mut_slice(&mut quo_coeffs_vec2);
        let mut rem_coeffs_vec2 = vec![Self::Field::zero(); rem_size];
        let rem_coeffs = HostSlice::from_mut_slice(&mut rem_coeffs_vec2);

        let vec_ops_cfg = VecOpsConfig::default();
        //vec_ops_cfg.batch_size = self.x_size as i32;
        ScalarCfg::transpose(quo_coeffs_tr, quo_x_size as u32, quo_y_size as u32, quo_coeffs, &vec_ops_cfg).unwrap();
        ScalarCfg::transpose(rem_coeffs_tr, rem_x_size as u32, rem_y_size as u32, rem_coeffs, &vec_ops_cfg).unwrap();
        (DensePolynomialExtEP::from_coeffs(quo_coeffs, quo_x_size, quo_y_size), DensePolynomialExtEP::from_coeffs(rem_coeffs, rem_x_size, rem_y_size))
    }

    fn div_by_vanishing(&self, denom_x_degree: i64, denom_y_degree: i64) -> (Self, Self) {
        if !( (denom_x_degree as usize).is_power_of_two() && (denom_y_degree as usize).is_power_of_two() ) {
            panic!("The denominators must have degress as powers of two.")
        }
        let numer_x_size = self.x_size;
        let numer_y_size = self.y_size;
        let numer_x_degree = self.x_degree;
        let numer_y_degree = self.y_degree;
        if numer_x_degree < denom_x_degree || numer_y_degree < denom_y_degree {
            panic!("The numerator must have grater degrees than denominators.")
        }
        // Assume that self's sizes are powers of two and optimized.
        let m = numer_x_size / denom_x_degree as usize;
        let n = numer_y_size / denom_y_degree as usize;
        let c = denom_x_degree as usize;
        let d = denom_y_degree as usize;
        
        let zeta = Self::FieldConfig::generate_random(1)[0];
        let xi = zeta;
        let vec_ops_cfg: VecOpsConfig = VecOpsConfig::default();
        if m>=2 && n== 2 {
            // A faster method for n==2, but not cover n>2.
            let block = vec![Self::Field::zero(); c * d];
            let mut blocks = vec![block; m * n];
            self._slice_coeffs_into_blocks(m, n, &mut blocks);
            
            // Computing A' (accumulation of blocks of the numerator)
            let mut scaled_acc_block_vec = vec![Self::Field::zero(); c * d];
            let xi_d = xi.pow(d);
            let mut acc_xi_d = Self::Field::one();
            
            for i in 0..n {
                let mut sub_acc_block_vec = vec![Self::Field::zero(); c * d];
                let sub_acc_block = unsafe{DeviceSlice::from_mut_slice(&mut sub_acc_block_vec)};
                
                // Accumulate blocks for current i
                for j in 0..m {
                    Self::FieldConfig::accumulate(
                        sub_acc_block, 
                        HostSlice::from_slice(&blocks[j*m + i]), 
                        &vec_ops_cfg
                    ).unwrap();
                }
                
                // Convert DeviceSlice to HostSlice
                let mut host_sub_acc_block_vec = vec![Self::Field::zero(); c * d];
                sub_acc_block.copy_to_host(&mut HostSlice::from_mut_slice(&mut host_sub_acc_block_vec)).unwrap();
                
                // Prepare result for scalar multiplication
                let mut scaled_result_vec = vec![Self::Field::zero(); c * d];
                
                // Define scalar_mul program
                let scalar_mul_program = FieldProgram::new(
                    |vars: &mut Vec<FieldSymbol>| {
                        // Multiply each element by the scalar
                        vars.push(vars[0] * acc_xi_d);
                    },
                    1 // 입력 값만 파라미터로 받음
                ).unwrap();
                
                // Setup parameters for execute_program
                let mut parameters = vec![
                    HostSlice::from_slice(&host_sub_acc_block_vec),
                    HostSlice::from_mut_slice(&mut scaled_result_vec)
                ];
                
                // Execute the scalar multiplication program
                execute_program(&mut parameters, &scalar_mul_program, &vec_ops_cfg).unwrap();
                
                // Accumulate results to scaled_acc_block_vec
                for idx in 0..c*d {
                    scaled_acc_block_vec[idx] = scaled_acc_block_vec[idx] + scaled_result_vec[idx];
                }
    
                acc_xi_d = acc_xi_d * xi_d;
            }
            
            // let acc_block_poly = DensePolynomialExt::from_coeffs(HostSlice::from_slice(&scaled_acc_block_vec), c, d);
            
            let scaled_acc_block = HostSlice::from_slice(&scaled_acc_block_vec);
            let acc_block_poly = DensePolynomialExt::from_coeffs(scaled_acc_block, c, d);
            let mut acc_block_eval = DeviceVec::<Self::Field>::device_malloc(c * d).unwrap();
            acc_block_poly.to_rou_evals(None, Some(&xi), &mut acc_block_eval);
            
            // Convert to HostSlice
            let mut host_acc_block_eval_vec = vec![Self::Field::zero(); c * d];
            acc_block_eval.copy_to_host(&mut HostSlice::from_mut_slice(&mut host_acc_block_eval_vec)).unwrap();
            
            // Computing Q_Z_tilde (eval of quo_y on rou-X and coset-Y)
            let _denom_val = (xi_d - Self::Field::one()).inv();
            let _denom_vec = vec![_denom_val; c * d];
            let mut quo_y_eval_vec = vec![Self::Field::zero(); c * d];
            
            // Define multiplication by inverse program
            let mul_program = FieldProgram::new(
                |vars: &mut Vec<FieldSymbol>| {
                    // Multiply by inverse of denominator
                    vars.push(vars[0] * _denom_val);
                },
                1 // 입력 값만 파라미터로 받음
            ).unwrap();
            
            // Setup parameters
            let mut parameters = vec![
                HostSlice::from_slice(&host_acc_block_eval_vec),
                HostSlice::from_mut_slice(&mut quo_y_eval_vec)
            ];
            
            // Execute the multiplication program
            execute_program(&mut parameters, &mul_program, &vec_ops_cfg).unwrap();
            
            // Convert back to DeviceVec
            let mut quo_y_eval_device = DeviceVec::<Self::Field>::device_malloc(c * d).unwrap();
            quo_y_eval_device.copy_from_host(&HostSlice::from_slice(&quo_y_eval_vec)).unwrap();
            
            // Computing Q_Z (quo_y polynomial)
            let quo_y = DensePolynomialExtEP::from_rou_evals(&quo_y_eval_device, c, d, None, Some(&xi));
            
            // Computing R = quo_y * (y^d - 1)
            let mut rem_x = &quo_y.mul_monomial(0, d) - &quo_y;
            rem_x.resize(m*c, n*d);
            
            // Computing B = quo_x * (x^c - 1)
            let lhs = self - &rem_x;
            
            // Computing B' (accumulation of blocks of B)
            let block = vec![Self::Field::zero(); c * (n*d)];
            let mut blocks = vec![block; m];
            lhs._slice_coeffs_into_blocks(m, 1, &mut blocks);
    
            let mut scaled_acc_block_vec = vec![Self::Field::zero(); c * (n*d)];

            let zeta_c = zeta.pow(c);
            let mut acc_zeta_c = Self::Field::one();
            
            for i in 0..m {
                // Scalar multiplication with acc_zeta_c
                let mut scaled_result_vec = vec![Self::Field::zero(); c * (n*d)];
                
                // Define scalar multiplication program with current acc_zeta_c
                let scalar_mul_program = FieldProgram::new(
                    |vars: &mut Vec<FieldSymbol>| {
                        // Multiply by current acc_zeta_c
                        vars.push(vars[0] * acc_zeta_c);
                    },
                    1 // 입력 값만 파라미터로 받음
                ).unwrap();
                
                // Setup parameters
                let mut parameters = vec![
                    HostSlice::from_slice(&blocks[i]),
                    HostSlice::from_mut_slice(&mut scaled_result_vec)
                ];
                
                // Execute scalar multiplication
                execute_program(&mut parameters, &scalar_mul_program, &vec_ops_cfg).unwrap();
                
                // Accumulate results
                for idx in 0..c*(n*d) {
                    scaled_acc_block_vec[idx] = scaled_acc_block_vec[idx] + scaled_result_vec[idx];
                }

                acc_zeta_c = acc_zeta_c * zeta_c;
            }
            
            let scaled_acc_block = HostSlice::from_slice(&scaled_acc_block_vec);
            let acc_block_poly = DensePolynomialExt::from_coeffs(scaled_acc_block, c, n*d);
            
            // Computing B_tilde (eval of B' on coset-X and rou-Y)
            let mut acc_block_eval = DeviceVec::device_malloc(c * (n*d)).unwrap();
            acc_block_poly.to_rou_evals(Some(&zeta), None, &mut acc_block_eval);
            
            // Convert to HostSlice
            let mut host_acc_block_eval_vec = vec![Self::Field::zero(); c * (n*d)];
            acc_block_eval.copy_to_host(&mut HostSlice::from_mut_slice(&mut host_acc_block_eval_vec)).unwrap();
            
            // Prepare division (multiplication by inverse)
            let _denom_val = (zeta_c - Self::Field::one()).inv();
            let mut quo_x_eval_vec = vec![Self::Field::zero(); c * (n*d)];
            
            // Define multiplication by inverse program
            let mul_inv_program = FieldProgram::new(
                |vars: &mut Vec<FieldSymbol>| {
                    // Multiply by inverse of denominator
                    vars.push(vars[0] * _denom_val);
                },
                1 // 입력 값만 파라미터로 받음
            ).unwrap();
            
            // Setup parameters
            let mut parameters = vec![
                HostSlice::from_slice(&host_acc_block_eval_vec),
                HostSlice::from_mut_slice(&mut quo_x_eval_vec)
            ];
            
            // Execute division program
            execute_program(&mut parameters, &mul_inv_program, &vec_ops_cfg).unwrap();
            
            // Convert to DeviceVec
            let mut quo_x_eval_device = DeviceVec::<Self::Field>::device_malloc(c * (n*d)).unwrap();
            quo_x_eval_device.copy_from_host(&HostSlice::from_slice(&quo_x_eval_vec)).unwrap();
            
            // Computing Q_Y (quo_x)
            let quo_x_ext = DensePolynomialExt::from_rou_evals(&quo_x_eval_device, c, n*d, Some(&zeta), None);
            let quo_x = DensePolynomialExtEP {
                poly: quo_x_ext.poly,
                x_degree: (c - 1) as i64,
                y_degree: (n*d - 1) as i64,
                x_size: c,
                y_size: n*d,
            };

            // Return the quotients
            (quo_x, quo_y)
        } else {
            panic!("div_by_vanishing currently does not support this numerator (x_degree is too large). Use divide_x and divide_y, instead.");
        }
    }

    fn div_by_ruffini(&self, x: Self::Field, y: Self:: Field) -> (Self, Self, Self::Field) where Self: Sized {
        // P(X,Y) = Q_Y(X,Y)(Y-y) + R_Y(X)
        // R_Y(X) = Q_X(X)(X-x) + R_X
        
        // Lengths of coeffs of P
        let x_len = self.x_size;
        let y_len = self.y_size;

        // Step 1: Extract the coefficients of univariate polynomials in Y for each X-degree
        // P(X,Y) = X^{deg-1} (P_{deg-1}(Y)) + X^{deg-2} (P_{deg-2}(Y)) + ... + X^{0} (P_{0}(Y))
        let mut p_i_coeffs_iter = vec![vec![Self::Field::zero();y_len]; x_len];
        for i in 0..x_len as u64 {
            let mut temp_vec = vec![Self::Field::zero(); y_len];
            let temp_buf = HostSlice::from_mut_slice(&mut temp_vec);
            self.get_univariate_polynomial_y(i).copy_coeffs(0, temp_buf);
            p_i_coeffs_iter[i as usize] = temp_vec;
        }
        
        // Step 2: Divide each polynomial P_i(Y) by (Y-y).
        let (q_y_coeffs_vec, r_y_coeffs_vec): (Vec<_>, Vec<_>) =  p_i_coeffs_iter
            .into_par_iter()
            .map(|coeffs| {
                let (q_i_y, r_i) = _div_uni_coeffs_by_ruffini(&coeffs, y);
                (q_i_y, r_i)
            })
            .unzip();

        // // let mut q_y_coeffs_vec = vec![Self::Field::zero(); x_len * y_len];      
        // // let mut r_y_coeffs_vec = vec![Self::Field::zero(); x_len];
        // for i in 0..(x_deg + 1) as usize {
        //     let (q_i_y, r_i) = uni_polys_iter[i].div_uni_by_ruffini(y);
        //     let mut q_i_y_coeffs_vec = vec![Self::Field::zero(); y_len];
        //     let q_i_y_coeffs = HostSlice::from_mut_slice(&mut q_i_y_coeffs_vec);
        //     q_i_y.copy_coeffs(0, q_i_y_coeffs);
        //     q_y_coeffs_vec[i * y_len..(i+1)*y_len].copy_from_slice(&q_i_y_coeffs_vec);
        //     r_y_coeffs_vec[i] = r_i
        // }   

        // Flatten q_y_coeffs_vec
        let q_y_coeffs_vec_flat: Vec<_> = q_y_coeffs_vec.into_par_iter().flatten().collect();
        let q_y_coeff_transpose = HostSlice::from_slice(&q_y_coeffs_vec_flat);
        let mut q_y_coeffs = DeviceVec::<Self::Field>::device_malloc(x_len * y_len).unwrap();
        let vec_ops_cfg = VecOpsConfig::default();
        ScalarCfg::transpose(q_y_coeff_transpose, x_len as u32, y_len as u32, &mut q_y_coeffs, &vec_ops_cfg).unwrap();

        let q_y = DensePolynomialExtEP::from_coeffs(&q_y_coeffs, x_len, y_len);

        // Divide R_Y(X) by (X-x).
        let (q_x_coeffs_vec, r_x) = _div_uni_coeffs_by_ruffini(&r_y_coeffs_vec, x);
        let q_x = DensePolynomialExtEP::from_coeffs(HostSlice::from_slice(&q_x_coeffs_vec), x_len, 1);
        (q_x, q_y, r_x)
    }

}

fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1 << (usize::BITS - (n - 1).leading_zeros())
}