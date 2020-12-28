use crate::extractor::ParallelExtractor;
use crate::features::FeatureId;
use crate::organisms::Organism;
use crate::positions;
use crate::samples;
use crate::types::{CompleteType, SpliceSiteType};
use crate::utils::load_scaffolds;
use anyhow::{Context, Result};
use ncrs::data::{Scaffold, Symbol};
use rand::prelude::*;
use std::collections::HashMap;

use std::path::Path;

// Number of upstream and downstream symbols to keep in the generated training
// sample.
const MARGIN: usize = 200;

pub struct Extractor<'a> {
    organism_csv: &'a Path,
    fasta_dir: &'a Path,
    positions_dir: &'a Path,
    target_dir: &'a Path,
    max_samples_per_category: usize,
}

impl<'a> Extractor<'a> {
    /// # Arguments
    ///
    /// * `max_samples_per_category` - maximum number of samples per organism
    ///   splice-site-type and positive/negative category. The actual number of
    ///   generated samples might be lower if there is not enough DNA data to
    ///   generate the desired number. For example, if this is set to 1000,
    ///   then there will be up to 1000 positive, donor examples for organism
    ///   `Aaoar1`.
    pub fn new(
        organism_csv: &'a Path,
        fasta_dir: &'a Path,
        positions_dir: &'a Path,
        target_dir: &'a Path,
        max_samples_per_category: usize,
    ) -> Self {
        Self {
            organism_csv,
            fasta_dir,
            positions_dir,
            target_dir,
            max_samples_per_category,
        }
    }

    fn extract_type(
        &self,
        organism: &Organism,
        complete_type: CompleteType,
        scaffolds: &HashMap<String, Scaffold>,
        mut rng: &mut ThreadRng,
    ) -> Result<()> {
        let mut reader = positions::io::Reader::from_dir(
            self.positions_dir,
            organism.organism_id(),
            complete_type,
        )?;

        let mut positions: Vec<(&str, &[Symbol], usize, FeatureId)> = Vec::new();
        for (scaffold_name, scaffold) in scaffolds.iter() {
            let sequence = scaffold.sequence();
            let scaffold_positions =
                reader
                    .read_scaffold_positions(scaffold_name)
                    .with_context(|| {
                        format!(
                            "Error while loading splice-site positions {} of organism {}",
                            complete_type,
                            organism.organism_id()
                        )
                    })?;

            let scaffold_feature_ids = reader
                .read_scaffold_feature_ids(scaffold_name)
                .with_context(|| {
                    format!(
                        "Error while loading parent feature IDs of splice-site {} of organism {}",
                        complete_type,
                        organism.organism_id()
                    )
                })?;

            positions.extend(
                scaffold_positions
                    .iter()
                    .zip(scaffold_feature_ids.iter())
                    .filter(|(p, _id)| **p > MARGIN && (sequence.len() - **p) > MARGIN)
                    .map(|(p, id)| (scaffold_name.as_str(), sequence, *p, *id)),
            );
        }

        positions.shuffle(&mut rng);
        positions.truncate(self.max_samples_per_category);

        let mut writer =
            samples::io::OrganismWriter::with_target_dir(self.target_dir, organism.organism_id())?;

        for (scaffold_name, sequence, position, feature_id) in positions {
            writer.write(
                &sequence[position - MARGIN..position + MARGIN],
                complete_type.is_positive(),
                &format!("{}", complete_type.splice_site_type()),
                MARGIN,
                position,
                Some(feature_id),
                scaffold_name,
            )?;
        }

        Ok(())
    }
}

impl<'a> ParallelExtractor for Extractor<'a> {
    fn organism_csv(&self) -> &Path {
        self.organism_csv
    }

    fn extract_organism(&self, organism: &Organism) -> Result<()> {
        let types = vec![
            CompleteType::new(SpliceSiteType::Donor, true),
            CompleteType::new(SpliceSiteType::Acceptor, true),
            CompleteType::new(SpliceSiteType::Donor, false),
            CompleteType::new(SpliceSiteType::Acceptor, false),
        ];

        let mut rng = rand::thread_rng();

        let scaffolds = load_scaffolds(self.fasta_dir, &organism)?;
        for complete_type in types.iter().copied() {
            self.extract_type(&organism, complete_type, &scaffolds, &mut rng)?;
        }

        Ok(())
    }
}
