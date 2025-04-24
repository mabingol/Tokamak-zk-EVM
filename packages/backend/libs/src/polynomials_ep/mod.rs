use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_core::traits::{Arithmetic, FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_bls12_381::symbol::bls12_381::FieldSymbol;
use icicle_core::ntt::{self, NTTConfig, NTTDir, NTTInitDomainConfig};
// use icicle_core::symbol::Symbol;
// Remove local enum definition since we want to use the one from icicle_core
// use icicle_core::program::PreDefinedProgram;
use icicle_core::program::{Instruction, Program, ReturningValueProgram};
use icicle_bls12_381::program::bls12_381::{FieldProgram, FieldReturningValueProgram};
use icicle_core::vec_ops::{VecOps, VecOpsConfig, execute_program, transpose_matrix};
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice, DeviceSlice, DeviceVec};
use std::{
    cmp,
    ops::{Add, AddAssign, Mul, Sub, Neg},
};
use super::vector_operations::{*};
use rayon::prelude::*;

// use crate::polynomials::{BivariatePolynomial, DensePolynomialExt};

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

    fn scale_coeffs_x(&self, scaler: Self::Field) -> Self;
    fn scale_coeffs_y(&self, scaler: Self::Field) -> Self;
    fn _scale_coeffs(&self, scaler: Self::Field, y_dir: bool, scaled_coeffs: &mut HostSlice<Self::Field>);

    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, x_size: usize, y_size: usize, coset_x: Option<&Self::Field>, coset_y: Option<&Self::Field>) -> Self;
    // Method to evaluate the polynomial over the roots-of-unity domain for power-of-two sized domain
    fn to_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, coset_x: Option<&Self::Field>, coset_y: Option<&Self::Field>, evals: &mut S);
    
    fn update_degree(&mut self);

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

}

impl BivariatePolynomialEP for DensePolynomialExtEP {
    type Field = ScalarField;
    type FieldConfig = ScalarCfg;

    fn update_degree(&mut self) {
        let size = self.x_size * self.y_size;
        let mut buf = vec![ScalarField::zero(); size];
        {
            let mut slice = HostSlice::from_mut_slice(&mut buf);
            self.poly.copy_coeffs(0, slice);
        }
    
        let x_deg = (0..self.x_size)
            .into_par_iter()
            .filter(|&i| {
                let row = &buf[i * self.y_size .. (i+1) * self.y_size];
                row.iter().any(|c| *c != ScalarField::zero())
            })
            .max()
            .unwrap_or(0);
    
        let y_deg = (0..self.y_size)
            .into_par_iter()
            .filter(|&j| {
                (0..self.x_size).any(|i| buf[i * self.y_size + j] != ScalarField::zero())
            })
            .max()
            .unwrap_or(0);
    
        self.x_degree = x_deg as i64;
        self.y_degree = y_deg as i64;
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

    fn scale_coeffs_x(&self, factor: ScalarField) -> Self {
        let x = self.x_size;
        let y = self.y_size;
        let total = x * y;

        // 1) 호스트에 기존 계수 통째로 복사
        let mut host_coeffs = vec![ScalarField::zero(); total];
        {
            let mut host_slice = HostSlice::from_mut_slice(&mut host_coeffs);
            self.copy_coeffs(0, host_slice);
        }

        // 2) 각 row(i)마다 factor^i 를 계산해 두는 배열
        let mut row_powers = vec![ScalarField::one(); x];
        for i in 1..x {
            row_powers[i] = row_powers[i - 1].mul(factor);
        }

        // 3) Rayon 병렬로 모든 계수에 곱해 주기
        let mut scaled = vec![ScalarField::zero(); total];
        scaled
            .par_chunks_mut(y)
            .enumerate()
            .for_each(|(i, chunk)| {
                let p = row_powers[i];
                let base = i * y;
                for j in 0..chunk.len() {
                    // chunk[j] = host_coeffs[base + j] * p
                    chunk[j] = host_coeffs[base + j].mul(p);
                }
            });

        // 4) 한 번에 Device 로 올려서 DensePolynomial 생성
        DensePolynomialExtEP::from_coeffs(
            HostSlice::from_slice(&scaled),
            x,
            y,
        )
    }

    fn scale_coeffs_y(&self, factor: ScalarField) -> Self {
        let x = self.x_size;
        let y = self.y_size;
        let total = x * y;

        let mut host_coeffs = vec![ScalarField::zero(); total];
        {
            let mut host_slice = HostSlice::from_mut_slice(&mut host_coeffs);
            self.copy_coeffs(0, host_slice);
        }

        let mut col_powers = vec![ScalarField::one(); y];
        for j in 1..y {
            col_powers[j] = col_powers[j - 1].mul(factor);
        }

        let mut scaled = vec![ScalarField::zero(); total];
        scaled
            .par_chunks_mut(y)
            .enumerate()
            .for_each(|(i, chunk)| {
                for j in 0..chunk.len() {
                    let base = i * y + j;
                    chunk[j] = host_coeffs[base].mul(col_powers[j]);
                }
            });

        DensePolynomialExtEP::from_coeffs(
            HostSlice::from_slice(&scaled),
            x,
            y,
        )
    }

    // fn scale_coeffs_x(&self, x_factor: Self::Field) -> Self {
    //     let mut scaled_coeffs_vec = vec![Self::Field::zero(); self.x_size * self.y_size];
    //     let scaled_coeffs = HostSlice::from_mut_slice(&mut scaled_coeffs_vec);
    //     self._scale_coeffs(x_factor, false, scaled_coeffs);
    //     return DensePolynomialExtEP::from_coeffs(
    //         scaled_coeffs,
    //         self.x_size, 
    //         self.y_size
    //     )
    // }

    // fn scale_coeffs_y(&self, y_factor: Self::Field) -> Self {
    //     let mut scaled_coeffs_vec = vec![Self::Field::zero(); self.x_size * self.y_size];
    //     let scaled_coeffs = HostSlice::from_mut_slice(&mut scaled_coeffs_vec);
    //     self._scale_coeffs(y_factor, true, scaled_coeffs);
    //     return DensePolynomialExtEP::from_coeffs(
    //         scaled_coeffs,
    //         self.x_size, 
    //         self.y_size
    //     )
    // }

    fn _scale_coeffs(
        &self,
        factor: Self::Field,
        y_dir: bool,
        scaled_coeffs: &mut HostSlice<Self::Field>,
    ) {
        let x_size = self.x_size;
        let y_size = self.y_size;
        let size   = x_size * y_size;
    
        // 1) 원본 계수를 DeviceVec에 복사
        let mut coeffs_dev = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
        self.poly.copy_coeffs(0, &mut coeffs_dev);
        
    
        // 2) 스케일 테이블 생성 → DeviceVec
        let mut scale_host = Vec::with_capacity(size);
        if !y_dir {
            let mut acc = Self::Field::one();
            for _ in 0..x_size {
                scale_host.extend(std::iter::repeat(acc.clone()).take(y_size));
                acc = acc * factor.clone();
            }
        } else {
            let mut acc = Self::Field::one();
            let mut row_scale = Vec::with_capacity(x_size);
            for _ in 0..x_size {
                row_scale.push(acc.clone());
                acc = acc * factor.clone();
            }
            for _ in 0..y_size {
                scale_host.extend(row_scale.iter().cloned());
            }
        }
        let mut scale_dev = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
        scale_dev
            .copy_from_host(HostSlice::from_slice(&scale_host))
            .unwrap();
    
        // 3) 결과를 담을 DeviceVec
        let mut result_dev = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
    
        // 4) (a, scale) → out 람다 프로그램 생성
        let mul_prog = FieldProgram::new(
            |params: &mut Vec<FieldSymbol>| {
                params[2] = params[0] * params[1];
            },
            3, // input a, input scale, output
        )
        .unwrap();
    
        // 5) execute_program 호출
        let mut cfg  = VecOpsConfig::default();
        let mut args: Vec<&dyn HostOrDeviceSlice<Self::Field>> = vec![
            &coeffs_dev, // &DeviceVec<T>
            &scale_dev,  // &DeviceVec<T>
            &mut result_dev, // &mut DeviceVec<T>
        ];
        execute_program(&mut args, &mul_prog, &cfg).unwrap();
    
        // 6) 결과를 호스트로 복사
        result_dev.copy_to_host(scaled_coeffs).unwrap();
    }
    
    
    
    

    // fn _scale_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(
    //     &self,
    //     factor: Self::Field,
    //     y_dir: bool,
    //     scaled_coeffs: &mut S,
    // ) {
    //     let x_size = self.x_size;
    //     let y_size = self.y_size;
    //     let size = x_size * y_size;
    
    //     // 1) Create a program to copy coefficients
    //     let coeffs_vec =&mut self.poly.coeffs_mut_slice();
    //     let mut host = vec![Self::Field::zero(); size];
    //     let mut host_slice = HostSlice::from_mut_slice(&mut host);
        
    //     // Copy program to transfer data from device to host
    //     let copy_program = FieldProgram::new(
    //         |vars: &mut Vec<FieldSymbol>| {
    //             vars.push(vars[0].clone());
    //         },
    //         1
    //     ).unwrap();
        
    //     // Execute copy operation
    //     let vec_ops_cfg = VecOpsConfig::default();
    //     let mut parameters = vec![
    //         coeffs_vec,
    //         host_slice.from_mut_slice(),
    //     ];
    //     execute_program(&mut parameters, &copy_program, &vec_ops_cfg).unwrap();
    
    //     // 2) 지수 테이블 준비
    //     if !y_dir {
    //         // X 방향: 각 row 에 factor^row
    //         let mut row_pows = Vec::with_capacity(x_size);
    //         let mut acc = Self::Field::one();
    //         for _ in 0..x_size {
    //             row_pows.push(acc.clone());
    //             acc = acc * factor.clone();
    //         }
    
    //         // 3) Rayon 으로 행 단위 병렬 스케일
    //         host.par_chunks_mut(y_size)
    //             .enumerate()
    //             .for_each(|(row, chunk)| {
    //                 let scale = &row_pows[row];
    //                 for v in chunk.iter_mut() {
    //                     *v = v.clone() * scale.clone();
    //                 }
    //             });
    //     } else {
    //         // Y 방향: 각 col 에 factor^col
    //         let mut col_pows = Vec::with_capacity(y_size);
    //         let mut acc = Self::Field::one();
    //         for _ in 0..y_size {
    //             col_pows.push(acc.clone());
    //             acc = acc * factor.clone();
    //         }
    
    //         // 3) Rayon 으로 요소별 병렬 스케일
    //         host.par_iter_mut()
    //             .enumerate()
    //             .for_each(|(i, v)| {
    //                 let col = i % y_size;
    //                 *v = v.clone() * col_pows[col].clone();
    //             });
    //     }
    
    //     // 4) 결과를 scaled_coeffs 에 복사
    //     let dst_ptr = unsafe { scaled_coeffs.as_mut_ptr() };
    //     let dst_len = scaled_coeffs.len();
    //     // 안전: ptr/len 은 항상 host.len() == size 와 일치
    //     let out_slice = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_len) };
    //     out_slice.copy_from_slice(&host);
    // }
    
    
    
    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, x_size: usize, y_size: usize, coset_x: Option<&Self::Field>, coset_y: Option<&Self::Field>) -> Self {
        if x_size == 0 || y_size == 0 {
            panic!("Invalid matrix size for from_rou_evals");
        }
        if x_size.is_power_of_two() == false || y_size.is_power_of_two() == false {
            panic!("The input sizes for from_rou_evals must be powers of two.")
        }

        let size = x_size * y_size;

        ntt::initialize_domain::<Self::Field>(
            ntt::get_root_of_unity::<Self::Field>(
                size.try_into()
                    .unwrap(),
            ),
            &ntt::NTTInitDomainConfig::default(),
        )
        .unwrap();

        let mut coeffs = DeviceVec::<Self::Field>::device_malloc(size).unwrap();
        let mut cfg = ntt::NTTConfig::<Self::Field>::default();
        
        // IFFT along X
        cfg.batch_size = y_size as i32;
        cfg.columns_batch = true;
        ntt::ntt(evals, ntt::NTTDir::kInverse, &cfg, &mut coeffs).unwrap();
        // IFFT along Y
        cfg.batch_size = x_size as i32;
        cfg.columns_batch = false;
        ntt::ntt_inplace(&mut coeffs, ntt::NTTDir::kInverse, &cfg).unwrap();

        ntt::release_domain::<Self::Field>().unwrap();

        let mut poly = DensePolynomialExtEP::from_coeffs(
            &coeffs,
            x_size,
            y_size
        );

        if let Some(_factor) = coset_x {
            let factor = _factor.inv();
            poly = poly.scale_coeffs_x(factor);
        }

        if let Some(_factor) = coset_y {
            let factor = _factor.inv();
            poly = poly.scale_coeffs_y(factor);
        }
        return poly
    }

    fn to_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, coset_x: Option<&Self::Field>, coset_y: Option<&Self::Field>, evals: &mut S) {
        let size = self.x_size * self.y_size;
        if evals.len() < size {
            panic!("Insufficient buffer length for to_rou_evals")
        }

        let mut scaled_poly = self.clone();

        if let Some(factor) = coset_x {
            scaled_poly = scaled_poly.scale_coeffs_x(*factor);
        }

        if let Some(factor) = coset_y {
            scaled_poly = scaled_poly.scale_coeffs_y(*factor);
        }

        let mut scaled_coeffs_vec = vec![Self::Field::zero(); self.x_size * self.y_size];
        let scaled_coeffs = HostSlice::from_mut_slice(&mut scaled_coeffs_vec);
        scaled_poly.copy_coeffs(0, scaled_coeffs);

        ntt::initialize_domain::<Self::Field>(
            ntt::get_root_of_unity::<Self::Field>(
                size.try_into()
                    .unwrap(),
            ),
            &ntt::NTTInitDomainConfig::default(),
        )
        .unwrap();
        let mut cfg = ntt::NTTConfig::<Self::Field>::default();
        // FFT along X
        cfg.batch_size = self.y_size as i32;
        cfg.columns_batch = true;
        ntt::ntt(scaled_coeffs, ntt::NTTDir::kForward, &cfg, evals).unwrap();
        
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
        let (new_x_size, new_y_size) = _find_size_as_twopower(target_x_size, target_y_size);
        if self.x_size == new_x_size && self.y_size == new_y_size {
            return;
        }
    
        let old_x = self.x_size;
        let old_y = self.y_size;
        let old_len = old_x * old_y;
        let new_len = new_x_size * new_y_size;
    
        // 1) 호스트로 기존 계수 복사
        let mut host_coeffs = vec![ScalarField::zero(); old_len];
        {
            let mut scratch = HostSlice::from_mut_slice(&mut host_coeffs);
            self.copy_coeffs(0, scratch);
        }
    
        // 2) Rayon 병렬로 각 행(row)별로 복사
        let mut new_host = vec![ScalarField::zero(); new_len];
        new_host
            .par_chunks_mut(new_y_size)
            .enumerate()
            .for_each(|(row_idx, row_chunk)| {
                if row_idx < old_x {
                    // 복사할 길이 계산
                    let take = old_y.min(new_y_size);
                    let src_start = row_idx * old_y;
                    // 안전하게 슬라이스
                    row_chunk[..take].copy_from_slice(&host_coeffs[src_start .. src_start + take]);
                }
                // else: 새로 생긴 행은 이미 0으로 초기화되어 있음
            });
    
        // 3) 한 번에 디바이스로 올리기
        let host_slice = HostSlice::from_slice(&new_host);
        self.poly = DensePolynomial::from_coeffs(host_slice, new_len);
    
        // 4) 사이즈 메타데이터 갱신
        self.x_size = new_x_size;
        self.y_size = new_y_size;
    }
     
    
    
    

    fn optimize_size(&mut self) {
        let target_x_size = self.x_degree as usize + 1;
        let target_y_size = self.y_degree as usize + 1;
        self.resize(target_x_size, target_y_size);
    }

    fn mul_monomial(&self, x_exp: usize, y_exp: usize) -> Self {
        // (0) 특별 케이스: x^0 y^0
        if x_exp == 0 && y_exp == 0 {
            return self.clone();
        }

        let x_size = self.x_size;
        let y_size = self.y_size;

        // (1) 원본 계수 한 번만 호스트에 복사
        let total = x_size * y_size;
        let mut orig_coeffs = vec![ScalarField::zero(); total];
        {
            let host_slice = HostSlice::from_mut_slice(&mut orig_coeffs);
            self.copy_coeffs(0, host_slice);
        }

        // (2) 출력 크기 결정 (power-of-two 올림)
        let target_x = (self.x_degree as usize) + x_exp + 1;
        let target_y = (self.y_degree as usize) + y_exp + 1;
        let new_x = next_pow2(target_x);
        let new_y = next_pow2(target_y);
        let new_total = new_x * new_y;

        // (3) 결과 버퍼 한 번만 할당
        let mut result = vec![ScalarField::zero(); new_total];

        // (4) 병렬 복사: 각 행(row) 단위로 처리
        result
            .par_chunks_mut(new_y)
            .enumerate()
            .for_each(|(new_i, row_chunk)| {
                // orig_row = new_i - x_exp
                if let Some(orig_i) = new_i.checked_sub(x_exp) {
                    if orig_i < x_size {
                        let src_start = orig_i * y_size;
                        let src = &orig_coeffs[src_start..src_start + y_size];
                        // 열 오프셋 y_exp 위치에 한 번에 복사
                        row_chunk[y_exp..y_exp + y_size].copy_from_slice(src);
                    }
                }
            });

        // (5) 한 번의 from_coeffs 호출로 디바이스 전송
        let host_slice = HostSlice::from_slice(&result);
        DensePolynomialExtEP::from_coeffs(host_slice, new_x, new_y)
    }

    // fn mul_monomial(&self, x_exp: usize, y_exp: usize) -> Self {
    //     // 1) 원본 크기, 목표 크기 계산 (비교를 위해 power-of-two 로 둡니다)
    //     let target_x = self.x_degree as usize + x_exp + 1;
    //     let target_y = self.y_degree as usize + y_exp + 1;
    //     let new_x_size = target_x.next_power_of_two();
    //     let new_y_size = target_y.next_power_of_two();
    //     let new_size = new_x_size * new_y_size;

    //     // 2) 호스트에서 “시프트된” 계수 벡터 만들기
    //     //    - 먼저 원본 계수를 host_coeffs 에 복사
    //     let old_size = self.x_size * self.y_size;
    //     let mut host_coeffs = vec![Self::Field::zero(); old_size];
    //     {
    //         let mut hs = HostSlice::from_mut_slice(&mut host_coeffs);
    //         self.copy_coeffs(0, hs);
    //     }
    //     //    - shift offset 계산 (1차원 인덱스)
    //     let offset = x_exp * new_y_size + y_exp;
    //     //    - 새로운 호스트 버퍼에 0 으로 채운 뒤, offset 부터 복사
    //     let mut shifted = vec![Self::Field::zero(); new_size];
    //     shifted[offset .. offset + old_size].copy_from_slice(&host_coeffs);

    //     // 3) 디바이스로 복사
    //     let mut padded_dev = DeviceVec::<Self::Field>::device_malloc(new_size).unwrap();
    //     padded_dev
    //         .copy_from_host(HostSlice::from_slice(&shifted))
    //         .unwrap();

    //     // 4) 결과를 받을 DeviceVec (초기값은 상관없음)
    //     let mut out_dev = DeviceVec::<Self::Field>::device_malloc(new_size).unwrap();

    //     // 5) “항등 λ” 프로그램 생성 (params[1] = params[0])
    //     let id_prog = FieldProgram::new(
    //         |vars: &mut Vec<FieldSymbol>| {
    //             vars[1] = vars[0];
    //         },
    //         2, // input, output
    //     )
    //     .unwrap();

    //     // 6) execute_program 으로 디바이스 상에서 복사 수행
    //     let mut args: Vec<&dyn HostOrDeviceSlice<Self::Field>> =
    //         vec![&padded_dev, &mut out_dev];
    //     let cfg = VecOpsConfig::default();
    //     execute_program(&mut args, &id_prog, &cfg).unwrap();

    //     // 7) 결과로 새 BivarPolynomialEP 구성
    //     DensePolynomialExtEP::from_coeffs(&out_dev, new_x_size, new_y_size)
    // }



    fn _mul(&self, rhs: &Self) -> Self {
        let (lhs_x_degree, lhs_y_degree) = self.degree();
        let (rhs_x_degree, rhs_y_degree) = rhs.degree();
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
        let target_x_size = self.x_size + rhs.x_size - 1;
        let target_y_size = self.y_size + rhs.y_size - 1;
        let mut lhs_ext = self.clone();
        let mut rhs_ext = rhs.clone();
        lhs_ext.resize(target_x_size, target_y_size);
        rhs_ext.resize(target_x_size, target_y_size);
        let x_size = lhs_ext.x_size;
        let y_size = lhs_ext.y_size;
        let extended_size = x_size * y_size;
        let cfg_vec_ops = VecOpsConfig::default();

        let mut lhs_evals = DeviceVec::<Self::Field>::device_malloc(extended_size).unwrap();
        let mut rhs_evals = DeviceVec::<Self::Field>::device_malloc(extended_size).unwrap();
        lhs_ext.to_rou_evals(None, None, &mut lhs_evals);
        rhs_ext.to_rou_evals(None, None, &mut rhs_evals);

        // Element-wise mult. of evaluations
        let mut out_evals = DeviceVec::<Self::Field>::device_malloc(extended_size).unwrap();
        ScalarCfg::mul(&lhs_evals, &rhs_evals, &mut out_evals, &cfg_vec_ops).unwrap();

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
            let acc_block_poly = DensePolynomialExtEP::from_coeffs(scaled_acc_block, c, d);
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
            let acc_block_poly = DensePolynomialExtEP::from_coeffs(scaled_acc_block, c, n*d);
            
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
            let quo_x_ext = DensePolynomialExtEP::from_rou_evals(&quo_x_eval_device, c, n*d, Some(&zeta), None);
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

    fn div_by_ruffini(&self, x: Self::Field, y: Self::Field) -> (Self, Self, Self::Field) where Self: Sized {
        // P(X,Y) = Q_Y(X,Y)(Y-y) + R_Y(X)
        // R_Y(X) = Q_X(X)(X-x) + R_X
        
        // Lengths of coeffs of P
        let x_len = self.x_size;
        let y_len = self.y_size;
    
        // Step 1: Extract the coefficients of univariate polynomials in Y for each X-degree
        let mut p_i_coeffs_iter = vec![vec![Self::Field::zero();y_len]; x_len];
        for i in 0..x_len as u64 {
            let mut temp_vec = vec![Self::Field::zero(); y_len];
            let temp_buf = HostSlice::from_mut_slice(&mut temp_vec);
            self.get_univariate_polynomial_y(i).copy_coeffs(0, temp_buf);
            p_i_coeffs_iter[i as usize] = temp_vec;
        }
        
        // Step 2: Create a program for Ruffini division
        // 루피니 나눗셈을 수행하는 프로그램 생성
        let ruffini_div_program = FieldProgram::new(
            |vars: &mut Vec<FieldSymbol>| {
                // 계수와 나누는 값을 가져옴
                let divisor = vars[0];  // 나누는 값 (x 또는 y)
                let coeff = vars[1];    // 현재 계수
                let prev_result = vars[2]; // 이전 결과
                
                // 새 결과 = 현재 계수 + 이전 결과 * 나누는 값
                vars[3] = coeff + prev_result * divisor;
            },
            4
        ).unwrap();
        
        let vec_ops_cfg = VecOpsConfig::default();
        
        // 각 X 차수에 대해 Y에 대한 루피니 나눗셈 수행
        let mut q_y_coeffs_vec = Vec::with_capacity(x_len);
        let mut r_y_coeffs_vec = Vec::with_capacity(x_len);
        
        for poly_coeffs in &p_i_coeffs_iter {
            let poly_degree = poly_coeffs.len() - 1;
            let mut quotient = vec![Self::Field::zero(); poly_degree];
            
            // 최고차항은 그대로 몫의 최고차항이 됨
            let mut remainder = poly_coeffs[poly_degree];
            if poly_degree > 0 {
                quotient[poly_degree - 1] = remainder;
            }
            
            // 나머지 항들에 대해 루피니 나눗셈 수행
            for i in (0..poly_degree).rev() {
                let mut result = Self::Field::zero();
                
                // Create longer-lived bindings for the slices
                let current_coeff = [poly_coeffs[i]];
                let divisor = [y];
                let prev_result = [remainder];
                
                // 프로그램 파라미터 설정
                let mut result_array = [result];
                let mut parameters = vec![
                    HostSlice::from_slice(&divisor),          // 나누는 값
                    HostSlice::from_slice(&current_coeff),    // 현재 계수
                    HostSlice::from_slice(&prev_result),      // 이전 결과
                    HostSlice::from_mut_slice(&mut result_array)  // 새 결과를 저장할 위치
                ];
                
                // execute_program 실행
                execute_program(&mut parameters, &ruffini_div_program, &vec_ops_cfg).unwrap();
                
                // 결과 저장
                remainder = result;
                if i > 0 {
                    quotient[i - 1] = remainder;
                }
            }
            
            q_y_coeffs_vec.push(quotient);
            r_y_coeffs_vec.push(remainder);
        }
        
        // 다음 2의 거듭제곱 크기 계산
        let next_pow2_y = (y_len - 1).next_power_of_two();
        
        // Flatten q_y_coeffs_vec 및 패딩 적용
        let mut q_y_coeffs_vec_flat = Vec::with_capacity(x_len * next_pow2_y);
        for q_y_coeff in q_y_coeffs_vec {
            let mut padded = q_y_coeff;
            padded.resize(next_pow2_y, Self::Field::zero());
            q_y_coeffs_vec_flat.extend(padded);
        }
        
        let q_y_coeff_transpose = HostSlice::from_slice(&q_y_coeffs_vec_flat);
        let mut q_y_coeffs = DeviceVec::<Self::Field>::device_malloc(x_len * next_pow2_y).unwrap();
        
        // 전치 연산 수행
        ScalarCfg::transpose(
            q_y_coeff_transpose, 
            x_len as u32, 
            next_pow2_y as u32, 
            &mut q_y_coeffs, 
            &vec_ops_cfg
        ).unwrap();
        
        // Q_Y 다항식 생성 - 2의 거듭제곱 크기 사용
        let q_y = DensePolynomialExtEP::from_coeffs(&q_y_coeffs, x_len, next_pow2_y);
        
        // R_Y(X)를 (X-x)로 나누기
        let r_y_degree = r_y_coeffs_vec.len() - 1;
        let mut q_x_coeffs = vec![Self::Field::zero(); r_y_degree];
        
        // 최고차항은 그대로 몫의 최고차항이 됨
        let mut r_x = r_y_coeffs_vec[r_y_degree];
        if r_y_degree > 0 {
            q_x_coeffs[r_y_degree - 1] = r_x;
        }
        
        // 나머지 항들에 대해 루피니 나눗셈 수행 - execute_program 사용
        for i in (0..r_y_degree).rev() {
            let mut result = Self::Field::zero();
            
            // 프로그램 파라미터 설정
            let current_coeff = [r_y_coeffs_vec[i]];
            let divisor = [x];
            let prev_result = [r_x];
            let mut result_slice = [result];
            let mut parameters = vec![
                HostSlice::from_slice(&divisor),        // 나누는 값
                HostSlice::from_slice(&current_coeff),  // 현재 계수
                HostSlice::from_slice(&prev_result),    // 이전 결과
                HostSlice::from_mut_slice(&mut result_slice)// 새 결과를 저장할 위치
            ];
            
            // execute_program 실행
            execute_program(&mut parameters, &ruffini_div_program, &vec_ops_cfg).unwrap();
            
            // 결과 저장
            r_x = result;
            if i > 0 {
                q_x_coeffs[i - 1] = r_x;
            }
        }
        
        // X 방향 다음 2의 거듭제곱 크기 계산
        let next_pow2_x = (x_len - 1).next_power_of_two();
        
        // 필요한 경우 패딩
        let mut padded_q_x = q_x_coeffs;
        padded_q_x.resize(next_pow2_x, Self::Field::zero());
        
        // Q_X 다항식 생성 - 2의 거듭제곱 크기 사용
        let q_x = DensePolynomialExtEP::from_coeffs(HostSlice::from_slice(&padded_q_x), next_pow2_x, 1);
        
        (q_x, q_y, r_x)
    }
    
    fn _div_uni_coeffs_by_ruffini(poly_coeffs_vec: &[ScalarField], x: ScalarField) -> (Vec<ScalarField>, ScalarField) {
        let len = poly_coeffs_vec.len();
        let mut q_coeffs_vec = std::vec![ScalarField::zero(); len];
        let mut b = poly_coeffs_vec[len - 1];
        q_coeffs_vec[len - 2] = b;
        for i in 3.. len + 1 {
            b = poly_coeffs_vec[len - i + 1] + b*x;
            q_coeffs_vec[len - i] = b;
        }
        let r = poly_coeffs_vec[0] + b*x;
        (q_coeffs_vec, r)
    }

}

fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1 << (usize::BITS - (n - 1).leading_zeros())
}

fn next_pow2(n: usize) -> usize {
    if n.is_power_of_two() { n } else { 1 << (usize::BITS - n.leading_zeros()) }
}