use crate::organisms::{load_organisms, Organism};
use anyhow::Result;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::path::Path;

/// This trait implements generic per-organism, parallel extractor of arbitrary
/// data (e.g. splice-site positions or training examples).
pub trait ParallelExtractor: Sync {
    /// Return a path to a CSV file with organism for which data should be
    /// extracted.
    fn organism_csv(&self) -> &Path;

    /// Extract data for a single organism.
    fn extract_organism(&self, organism: &Organism) -> Result<()>;

    /// Extract data for all organisms with some parallelization and show a
    /// progress bar.
    fn extract(&self) -> Result<()> {
        let organisms = load_organisms(self.organism_csv())?;

        let progress_bar = ProgressBar::new(organisms.len() as u64);
        progress_bar.enable_steady_tick(500);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {wide_bar:cyan/blue} {pos:>4}/{len:4} ({eta})"),
        );

        organisms
            .par_iter()
            .progress_with(progress_bar)
            .try_for_each(|o| self.extract_organism(o))
    }
}
