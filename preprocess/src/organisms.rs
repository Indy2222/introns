use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Organism {
    organism_id: String,
    fasta_file: String,
    gff_file: String,
    genome_size: u64,
    taxonomy_phylum: Option<String>,
    taxonomy_sub_phylum: Option<String>,
    taxonomy_class: Option<String>,
    taxonomy_sub_class: Option<String>,
    taxonomy_order: Option<String>,
    taxonomy_sub_order: Option<String>,
    taxonomy_family: Option<String>,
    taxonomy_sub_family: Option<String>,
    taxonomy_genus: Option<String>,
    taxonomy_species: Option<String>,
}

impl Organism {
    pub fn organism_id(&self) -> &str {
        self.organism_id.as_str()
    }

    pub fn fasta_file(&self) -> &str {
        self.fasta_file.as_str()
    }

    pub fn gff_file(&self) -> &str {
        self.gff_file.as_str()
    }
}

pub fn load_organisms(csv_file: &Path) -> Result<Vec<Organism>> {
    let file = File::open(csv_file)
        .with_context(|| format!("Failed to open organisms CSV {}", csv_file.display()))?;
    let mut reader = csv::Reader::from_reader(BufReader::new(file));
    reader
        .deserialize()
        .map(|r| r.context("Failed to read and parse CSV with organisms"))
        .collect()
}
