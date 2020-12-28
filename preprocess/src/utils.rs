use crate::organisms::Organism;
use anyhow::{Context, Result};
use ncrs::{data::Scaffold, fasta};
use std::collections::HashMap;
use std::path::Path;

pub fn load_scaffolds<P: AsRef<Path>>(
    fasta_dir: P,
    organism: &Organism,
) -> Result<HashMap<String, Scaffold>> {
    let mut fasta_file = fasta_dir.as_ref().to_path_buf();
    fasta_file.push(organism.fasta_file());

    let mut scaffolds = fasta::load_fasta(&fasta_file)
        .with_context(|| format!("Failed to load FASTA file {}", fasta_file.display()))?;

    let mut scaffolds_map = HashMap::with_capacity(scaffolds.len());
    for scaffold in scaffolds.drain(..) {
        scaffolds_map.insert(String::from(scaffold.name()), scaffold);
    }
    Ok(scaffolds_map)
}
