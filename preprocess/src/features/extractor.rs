use crate::extractor::ParallelExtractor;
use crate::features::{self, FeatureId, SequenceFeature};
use crate::organisms::Organism;
use anyhow::{Context, Result};
use ncrs::{
    data::{Annotation, Feature, Strand},
    gff,
};
use std::path::Path;

const PROTEIN_ID_ATTR: &str = "proteinId";

pub struct Extractor<'a> {
    organism_csv: &'a Path,
    gff_dir: &'a Path,
    target_dir: &'a Path,
}

impl<'a> Extractor<'a> {
    pub fn new(organism_csv: &'a Path, gff_dir: &'a Path, target_dir: &'a Path) -> Self {
        Self {
            organism_csv,
            gff_dir,
            target_dir,
        }
    }

    fn write_genes_and_introns(
        &self,
        organism_id: &str,
        coding_sequences: &[Annotation],
    ) -> Result<()> {
        let mut gene_writer =
            features::io::Writer::write_to_dir(self.target_dir, organism_id, "genes")
                .context("Failed to open gene writer")?;
        let mut intron_writer =
            features::io::Writer::write_to_dir(self.target_dir, organism_id, "introns")
                .context("Failed to open introns writer")?;

        let mut last_gene_start: Option<usize> = None;
        let mut last_exon_end = 0;
        let mut last_scaffold = String::new();
        let mut last_protein_id = !0;

        for cds in coding_sequences {
            let proteid_id = parse_protein_id(cds.attributes())?;

            if proteid_id != last_protein_id || cds.scaffold() != last_scaffold.as_str() {
                if let Some(last_gene_start) = last_gene_start {
                    gene_writer
                        .write(&SequenceFeature::new(
                            FeatureId::from_bytes(last_protein_id.to_be_bytes()),
                            None,
                            last_scaffold.to_owned(),
                            last_gene_start,
                            last_exon_end,
                        ))
                        .context("Faild to write gene")?;
                }

                last_gene_start = Some(cds.start());
                last_protein_id = proteid_id;
                last_scaffold = cds.scaffold().to_owned();
            } else {
                intron_writer
                    .write(&SequenceFeature::with_generated_id(
                        "intron",
                        Some(FeatureId::from_bytes(proteid_id.to_be_bytes())),
                        cds.scaffold().to_owned(),
                        last_exon_end,
                        cds.start(),
                    ))
                    .context("Failed to write intron")?;
            }

            last_exon_end = cds.end();
        }

        Ok(())
    }
}

impl<'a> ParallelExtractor for Extractor<'a> {
    fn organism_csv(&self) -> &Path {
        self.organism_csv
    }

    fn extract_organism(&self, organism: &Organism) -> Result<()> {
        let mut gff_file_path = self.gff_dir.to_path_buf();
        gff_file_path.push(organism.gff_file());

        let mut annotations = gff::load_gff_file(&gff_file_path).with_context(|| {
            format!(
                "Failed to load annotations of organism {}",
                organism.organism_id()
            )
        })?;

        let annotations: Vec<Annotation> = annotations
            .drain(..)
            .filter(|a| a.feature() == Feature::CDS && a.strand() == Strand::Positive)
            .collect();

        self.write_genes_and_introns(organism.organism_id(), &annotations)
    }
}

fn parse_protein_id(attributes: &str) -> Result<u128> {
    for attribute in attributes.split(';') {
        let mut split = attribute.trim().split_whitespace();
        if split.next() == Some(PROTEIN_ID_ATTR) {
            let protein_id: u128 = match split.next() {
                Some(protein_id) => protein_id
                    .parse()
                    .with_context(|| format!("Invalid `{}` attribute", PROTEIN_ID_ATTR))?,
                None => bail!("Invalid `{}` attribute", PROTEIN_ID_ATTR),
            };

            return Ok(protein_id);
        }
    }

    bail!("Missing attribute `{}`.", PROTEIN_ID_ATTR);
}
