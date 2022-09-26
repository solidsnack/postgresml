// fn main() {
//     let mut embeddings = [[0 as f32; 128]; 10_000];
//     for i in 0..10_000 {
//         for j in 0..128 {
//             embeddings[i][j] = rand::random()
//         }
//     };
//     // println!("{:?}", embeddings);
// }
//-- Generate 10,000 embeddings with 128 dimensions as FLOAT4[] type.
// CREATE TABLE embeddings AS 
// SELECT ARRAY_AGG(random())::FLOAT4[] AS vector
// FROM generate_series(1, 12800000) i
// GROUP BY i % 100000;

// WITH test AS (
//     SELECT ARRAY_AGG(random())::FLOAT4[] AS vector
//     FROM generate_series(1, 128) i
// )
// SELECT benchmark.dot_product_into(embeddings.vector, test.vector) AS dot_product
// FROM embeddings, test
// ORDER BY 1
// LIMIT 1; 

extern crate blas;
extern crate openblas_src;

use pgx::*;
extern crate linfa;
use crate::linfa::prelude::{Fit, Predict, ToConfusionMatrix};
use crate::linfa::dataset::Records;
use linfa_logistic::LogisticRegression;

pg_module_magic!();


#[pg_extern(immutable, parallel_safe, strict)]
fn dot_product_rust(vector: Vec<f32>, other: Vec<f32>) -> f32 {
    vector
        .as_slice()
        .iter()
        .zip(other.as_slice().iter())
        .map(|(a, b)| a * b )
        .sum()
}

#[pg_extern(immutable, parallel_safe, strict)]
fn dot_product_into(vector: Vec<f32>, other: Vec<f32>) -> f32 {
    vector
        .into_iter()
        .zip(other.into_iter())
        .map(|(a, b)| a * b )
        .sum()
}

#[pg_extern(immutable, parallel_safe, strict)]
fn dot_product_blas(vector: Vec<f32>, other: Vec<f32>) -> f32 {
    unsafe {
        blas::sdot(
            vector.len().try_into().unwrap(),
            vector.as_slice(),
            1,
            other.as_slice(),
            1,
        )
    }
}

use linfa::prelude::*;
use linfa_logistic::MultiLogisticRegression;

use std::error::Error;

#[pg_extern(immutable, parallel_safe, strict)]
fn test_linfa_logistic() -> f32 {
    let (train, valid) = linfa_datasets::iris().split_with_ratio(0.75);

    info!(
        "Fit Multinomial Logistic Regression classifier with #{} training points",
        train.nsamples()
    );

    // fit a Logistic regression model with 150 max iterations
    let model = MultiLogisticRegression::default()
        .max_iterations(300)
        .fit(&train)
        .unwrap();

    // predict and map targets
    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    info!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    info!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    1.
}

use linfa::composing::MultiClassModel;
use linfa::prelude::*;
use linfa_svm::{error::Result, Svm};

#[pg_extern(immutable, parallel_safe, strict)]
fn test_linfa_svm() -> f32 {
    let (train, valid) = linfa_datasets::iris().split_with_ratio(0.75);

    info!(
        "Fit SVM classifier with #{} training points",
        train.nsamples()
    );

    let params = Svm::<_, Pr>::params()
        //.pos_neg_weights(5000., 500.)
        .gaussian_kernel(30.0);

    let model = train
        .one_vs_all().unwrap()
        .into_iter()
        .map(|(l, x)| (l, params.fit(&x).unwrap()))
        .collect::<MultiClassModel<_, _>>();

    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // Print the confusion matrix
    info!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    info!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    1.
}