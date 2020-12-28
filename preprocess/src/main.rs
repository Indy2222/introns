#[macro_use]
extern crate anyhow;

#[macro_use]
extern crate clap;

mod extractor;
mod features;
mod introns;
mod organisms;
mod positions;
mod samples;
mod types;
mod utils;

use anyhow::{Context, Result};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use extractor::ParallelExtractor;
use std::path::Path;

fn main() {
    let features_cmd = SubCommand::with_name("features")
        .arg(
            Arg::with_name("organisms-csv")
                .short("o")
                .long("organisms-csv")
                .help("Path to a CSV files with organism DNA sequence annotations.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("gff-dir")
                .short("g")
                .long("gff-dir")
                .help(
                    "Path to a directory with DNA annotations in the form of \
                     GFF files.",
                )
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("target-dir")
                .short("t")
                .long("target-dir")
                .help(
                    "Path to a directory where CSV files with genes and \
                     introns will get stored.",
                )
                .takes_value(true)
                .required(true),
        );

    let positions_cmd = SubCommand::with_name("positions")
        .arg(
            Arg::with_name("organisms-csv")
                .short("o")
                .long("organisms-csv")
                .help("Path to a CSV files with organism DNA sequence annotations.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("fasta-dir")
                .short("f")
                .long("fasta-dir")
                .help("Path to a directory with organisms' FASTA files with DNA sequences.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("feature-dir")
                .short("g")
                .long("feature-dir")
                .help("Path to a directory with CSV files with gene and intron positions.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("target-dir")
                .short("t")
                .long("target-dir")
                .help("Path where CSV files with candidate and real splice-site positions will get stored.")
                .takes_value(true)
                .required(true),
        );

    let samples_cmd = SubCommand::with_name("samples")
        .arg(
            Arg::with_name("organisms-csv")
                .short("o")
                .long("organisms-csv")
                .help("Path to a CSV files with organism DNA sequence annotations.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("fasta-dir")
                .short("f")
                .long("fasta-dir")
                .help("Path to a directory with organisms' FASTA files with DNA sequences.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("positions-dir")
                .short("p")
                .long("positions-dir")
                .help("Path to a directory with CSV files with splice-site positions.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("target-dir")
                .short("t")
                .long("target-dir")
                .help("Path where CSV files with candidate and real splice-site positions will get stored.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("num-samples")
                .short("n")
                .long("num-samples")
                    .help(
                        "Number of samples per organism, per negative/postive \
                         and donor/acceptor category to be generated."
                    )
                .takes_value(true)
                .required(true),
        );

    let introns_cmd = SubCommand::with_name("introns")
        .about(
            "Create training NPZ files from a CSV file with candidate \
             introns. The input file has two columns: `sequence` (id est a \
             string of `A`, `C`, `T`, `G`, `N`) and `label`. Label is `-1` \
             for negative and `1` for positive example.",
        )
        .arg(
            Arg::with_name("introns-csv")
                .short("i")
                .long("introns-csv")
                .help("Path to a CSV file with candidate introns.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("target-dir")
                .short("t")
                .long("target-dir")
                .help("Path where generated sample NPZ files will be stored.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("placeholder-name")
                .short("n")
                .long("placeholder-name")
                .help(
                    "Source CSV file doesn't contain name or IDs of organisms \
                     so it is not known which DNA sample belongs to which \
                     organism. This placeholder name is used instead.",
                )
                .takes_value(true)
                .required(true),
        );

    let matches = App::new(crate_name!())
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .version(crate_version!())
        .about("Intron detection data preprocessing tool.")
        .long_about(include_str!("../description.txt"))
        .subcommand(features_cmd)
        .subcommand(positions_cmd)
        .subcommand(samples_cmd)
        .subcommand(introns_cmd)
        .get_matches();

    let result = match matches.subcommand() {
        ("features", Some(matches)) => subcommand_features(matches),
        ("positions", Some(matches)) => subcommand_positions(matches),
        ("samples", Some(matches)) => subcommand_samples(matches),
        ("introns", Some(matches)) => subcommand_introns(matches),
        _ => panic!("Unrecognized command."),
    };

    if let Err(error) = result {
        let message = error.chain().fold(String::new(), |message, item| {
            format!("{}: {}", message, item)
        });
        panic!("{}", message);
    }
}

fn subcommand_features(matches: &ArgMatches) -> Result<()> {
    let organism_csv = Path::new(matches.value_of("organisms-csv").unwrap());
    let gff_dir = Path::new(matches.value_of("gff-dir").unwrap());
    let target_dir = Path::new(matches.value_of("target-dir").unwrap());
    let extractor = features::extractor::Extractor::new(organism_csv, gff_dir, target_dir);
    extractor.extract()
}

fn subcommand_positions(matches: &ArgMatches) -> Result<()> {
    let organism_csv = Path::new(matches.value_of("organisms-csv").unwrap());
    let fasta_dir = Path::new(matches.value_of("fasta-dir").unwrap());
    let features_dir = Path::new(matches.value_of("feature-dir").unwrap());
    let target_dir = Path::new(matches.value_of("target-dir").unwrap());
    let extractor =
        positions::extractor::Extractor::new(organism_csv, fasta_dir, features_dir, target_dir);
    extractor.extract()
}

fn subcommand_samples(matches: &ArgMatches) -> Result<()> {
    let organism_csv = Path::new(matches.value_of("organisms-csv").unwrap());
    let fasta_dir = Path::new(matches.value_of("fasta-dir").unwrap());
    let positions_dir = Path::new(matches.value_of("positions-dir").unwrap());
    let target_dir = Path::new(matches.value_of("target-dir").unwrap());

    let num_samples = matches.value_of("num-samples").unwrap();
    let num_samples: usize = num_samples
        .parse()
        .context("Invalid 'num-samples' argument value")?;

    let extractor = samples::extractor::Extractor::new(
        organism_csv,
        fasta_dir,
        positions_dir,
        target_dir,
        num_samples,
    );
    extractor.extract()
}

fn subcommand_introns(matches: &ArgMatches) -> Result<()> {
    let introns_csv = Path::new(matches.value_of("introns-csv").unwrap());
    let target_dir = Path::new(matches.value_of("target-dir").unwrap());
    let placeholder_name = matches.value_of("placeholder-name").unwrap();

    let extractor = introns::extractor::Extractor::new(introns_csv, target_dir, placeholder_name);
    extractor.extract()
}
