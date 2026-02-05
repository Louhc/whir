use std::{sync::Arc, time::{Duration, Instant}};

use ark_ff::{FftField, Field};
use ark_serialize::CanonicalSerialize;
use clap::Parser;
use spongefish::{domain_separator, session, Codec};
use whir::{
    cmdline_utils::{AvailableFields, AvailableHash, WhirType},
    crypto::fields,
    hash::HASH_COUNTER,
    ntt::{RSDefault, ReedSolomon},
    parameters::{
        default_max_pow, FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType,
    },
    poly_utils::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    transcript::{codecs::Empty, ProverState, VerifierState},
    whir::{
        committer::CommitmentReader,
        statement::{Statement, Weights},
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 't', long = "type", default_value = "PCS")]
    protocol_type: WhirType,

    #[arg(short = 'l', long, default_value = "128")]
    security_level: usize,

    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    #[arg(short = 'm', long, default_value = "1")]
    iterations: usize,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(long = "linear_constraints", default_value = "0")]
    num_linear_constraints: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(long = "reps", default_value = "1")]
    verifier_repetitions: usize,

    #[arg(short = 'i', long = "initfold", default_value = "4")]
    first_round_folding_factor: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "ProvableList")]
    soundness_type: SoundnessType,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks3")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    hash: AvailableHash,
}

fn main() {
    let mut args = Args::parse();
    let field = args.field;

    if args.pow_bits.is_none() {
        args.pow_bits = Some(default_max_pow(args.num_variables, args.rate));
    }

    runner(&args, field);
}

fn runner(args: &Args, field: AvailableFields) {
    // Type reflection on field
    match field {
        AvailableFields::Goldilocks1 => run_whir::<fields::Field64>(args),
        AvailableFields::Goldilocks2 => run_whir::<fields::Field64_2>(args),
        AvailableFields::Goldilocks3 => run_whir::<fields::Field64_3>(args),
        AvailableFields::Field128 => run_whir::<fields::Field128>(args),
        AvailableFields::Field192 => run_whir::<fields::Field192>(args),
        AvailableFields::Field256 => run_whir::<fields::Field256>(args),
    }
}

fn run_whir<F>(args: &Args)
where
    F: FftField + CanonicalSerialize + Codec,
{
    let reed_solomon = Arc::new(RSDefault);
    let basefield_reed_solomon = reed_solomon.clone();

    match args.protocol_type {
        WhirType::PCS => {
            run_whir_pcs::<F>(args, reed_solomon, basefield_reed_solomon);
        }
        WhirType::LDT => {
            run_whir_as_ldt::<F>(args, reed_solomon, basefield_reed_solomon);
        }
    }
}

fn run_whir_as_ldt<F>(
    args: &Args,
    reed_solomon: Arc<dyn ReedSolomon<F>>,
    basefield_reed_solomon: Arc<dyn ReedSolomon<F::BasePrimeField>>,
) where
    F: FftField + CanonicalSerialize + Codec,
{
    use whir::whir::{
        committer::CommitmentWriter, parameters::WhirConfig, prover::Prover, verifier::Verifier,
    };

    // Runs as a LDT
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let iterations = args.iterations;
    let reps = iterations;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let soundness_type = args.soundness_type;
    let hash_id = args.hash.hash_id();

    if args.num_evaluations > 1 {
        println!("Warning: running as LDT but a number of evaluations to be proven was specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = ProtocolParameters {
        initial_statement: false,
        security_level,
        pow_bits,
        folding_factor: FoldingFactor::ConstantFromSecondRound(
            first_round_folding_factor,
            folding_factor,
        ),
        soundness_type,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        hash_id,
    };

    let params = WhirConfig::<F>::new(
        reed_solomon,
        basefield_reed_solomon,
        mv_params,
        &whir_params,
    );

    let ds = domain_separator!("🌪️")
        .session(session!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);


    println!("=========================================");
    println!("Whir (LDT) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    let polynomial = CoefficientList::new(
        (0..num_coeffs)
            .map(<F as Field>::BasePrimeField::from)
            .collect(),
    );

    let committer = CommitmentWriter::new(params.clone());
    let prover = Prover::new(params.clone());
    let statement = Statement::new(num_variables);

    let mut commit_times = Vec::with_capacity(iterations);
    let mut open_times = Vec::with_capacity(iterations);
    let mut proof_opt = None;
    for i in 0..iterations {
        let mut prover_state = ProverState::from(ds.std_prover());

        let commit_start = Instant::now();
        let witness = committer.commit(&mut prover_state, &polynomial);
        let commit_ms = commit_start.elapsed().as_secs_f64() * 1000.0;
        println!("ITER_{}_COMMIT_MS: {:.3}", i + 1, commit_ms);
        commit_times.push(commit_ms);

        let open_start = Instant::now();
        prover.prove(&mut prover_state, statement.clone(), witness);
        let open_ms = open_start.elapsed().as_secs_f64() * 1000.0;
        println!("ITER_{}_OPEN_MS: {:.3}", i + 1, open_ms);
        open_times.push(open_ms);

        proof_opt = Some(prover_state.proof());
    }

    let proof = proof_opt.unwrap();
    let proof_size = proof.narg_string.len() + proof.hints.len();
    let avg = |vals: &[f64]| vals.iter().sum::<f64>() / vals.len() as f64;
    let commit_ms = avg(&commit_times);
    let open_ms = avg(&open_times);
    let prover_ms = commit_ms + open_ms;
    println!("Prover time: {:.1?}", Duration::from_secs_f64(prover_ms / 1000.0));
    println!("Proof size: {:.1} KiB", proof_size as f64 / 1024.0);
    println!("COMMIT_TIME_MS: {:.3}", commit_ms);
    println!("OPEN_TIME_MS: {:.3}", open_ms);
    println!("PROVER_TIME_MS: {:.3}", prover_ms);
    println!("PROOF_SIZE_KB: {:.3}", proof_size as f64 / 1024.0);

    // Just not to count that initial inversion (which could be precomputed)
    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state =
            VerifierState::from(ds.std_verifier(&proof.narg_string), &proof.hints);

        let parsed_commitment = commitment_reader
            .parse_commitment(&mut verifier_state)
            .unwrap();
        verifier
            .verify(&mut verifier_state, &parsed_commitment, &statement)
            .unwrap();
    }
    let verify_avg = whir_verifier_time.elapsed() / reps as u32;
    let verify_ms = verify_avg.as_secs_f64() * 1000.0;
    dbg!(verify_avg);
    dbg!(HASH_COUNTER.get() as f64 / reps as f64);
    println!("VERIFY_TIME_MS: {:.3}", verify_ms);
}

#[allow(clippy::too_many_lines)]
fn run_whir_pcs<F>(
    args: &Args,
    reed_solomon: Arc<dyn ReedSolomon<F>>,
    basefield_reed_solomon: Arc<dyn ReedSolomon<F::BasePrimeField>>,
) where
    F: FftField + CanonicalSerialize + Codec,
{
    use whir::whir::{
        committer::CommitmentWriter, parameters::WhirConfig, prover::Prover, statement::Statement,
        verifier::Verifier,
    };

    // Runs as a PCS
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let iterations = args.iterations;
    let reps = iterations;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let soundness_type = args.soundness_type;
    let num_evaluations = args.num_evaluations;
    let num_linear_constraints = args.num_linear_constraints;
    let hash_id = args.hash.hash_id();

    if num_evaluations == 0 {
        println!("Warning: running as PCS but no evaluations specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = ProtocolParameters {
        initial_statement: true,
        security_level,
        pow_bits,
        folding_factor: FoldingFactor::ConstantFromSecondRound(
            first_round_folding_factor,
            folding_factor,
        ),
        soundness_type,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        hash_id,
    };

    let params = WhirConfig::<F>::new(
        reed_solomon,
        basefield_reed_solomon,
        mv_params,
        &whir_params,
    );

    let ds = domain_separator!("🌪️")
        .session(session!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);
    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    let polynomial = CoefficientList::new(
        (0..num_coeffs)
            .map(<F as Field>::BasePrimeField::from)
            .collect(),
    );
    let mut statement: Statement<F> = Statement::<F>::new(num_variables);
    // Evaluation constraint
    let points: Vec<_> = (0..num_evaluations)
        .map(|x| MultilinearPoint(vec![F::from(x as u64); num_variables]))
        .collect();

    for point in &points {
        let eval = polynomial.evaluate_at_extension(point);
        let weights = Weights::evaluation(point.clone());
        statement.add_constraint(weights, eval);
    }

    // Linear constraint
    for _ in 0..num_linear_constraints {
        let input = CoefficientList::new((0..num_coeffs).map(F::from).collect());
        let input: EvaluationsList<F> = input.clone().into();

        let linear_claim_weight = Weights::linear(input.clone());
        let poly = EvaluationsList::from(polynomial.clone().to_extension());

        let sum = linear_claim_weight.weighted_sum(&poly);
        statement.add_constraint(linear_claim_weight, sum);
    }

    let committer = CommitmentWriter::new(params.clone());
    let prover = Prover::new(params.clone());
    let mut commit_times = Vec::with_capacity(iterations);
    let mut open_times = Vec::with_capacity(iterations);
    let mut proof_opt = None;
    for i in 0..iterations {
        let mut prover_state = ProverState::from(ds.std_prover());

        let commit_start = Instant::now();
        let witness = committer.commit(&mut prover_state, &polynomial);
        let commit_ms = commit_start.elapsed().as_secs_f64() * 1000.0;
        println!("ITER_{}_COMMIT_MS: {:.3}", i + 1, commit_ms);
        commit_times.push(commit_ms);

        let open_start = Instant::now();
        prover.prove(&mut prover_state, statement.clone(), witness);
        let open_ms = open_start.elapsed().as_secs_f64() * 1000.0;
        println!("ITER_{}_OPEN_MS: {:.3}", i + 1, open_ms);
        open_times.push(open_ms);

        proof_opt = Some(prover_state.proof());
    }

    let proof = proof_opt.unwrap();
    let avg = |vals: &[f64]| vals.iter().sum::<f64>() / vals.len() as f64;
    let commit_ms = avg(&commit_times);
    let open_ms = avg(&open_times);
    let prover_ms = commit_ms + open_ms;
    let proof_kb = (proof.narg_string.len() + proof.hints.len()) as f64 / 1024.0;
    println!("Prover time: {:.1?}", Duration::from_secs_f64(prover_ms / 1000.0));
    println!("Proof size: {:.1} KiB", proof_kb);
    println!("COMMIT_TIME_MS: {:.3}", commit_ms);
    println!("OPEN_TIME_MS: {:.3}", open_ms);
    println!("PROVER_TIME_MS: {:.3}", prover_ms);
    println!("PROOF_SIZE_KB: {:.3}", proof_kb);

    // Just not to count that initial inversion (which could be precomputed)
    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state =
            VerifierState::from(ds.std_verifier(&proof.narg_string), &proof.hints);

        let parsed_commitment = commitment_reader
            .parse_commitment(&mut verifier_state)
            .unwrap();
        verifier
            .verify(&mut verifier_state, &parsed_commitment, &statement)
            .unwrap();
    }
    let verify_avg = whir_verifier_time.elapsed() / reps as u32;
    let verify_ms = verify_avg.as_secs_f64() * 1000.0;
    println!(
        "Verifier time: {:.1?}",
        verify_avg
    );
    println!("VERIFY_TIME_MS: {:.3}", verify_ms);
    println!(
        "Average hashes: {:.1}k",
        (HASH_COUNTER.get() as f64 / reps as f64) / 1000.0
    );
}
