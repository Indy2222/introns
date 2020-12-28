use crate::introns::csv::CandidateIntronCsvReader;
use crate::samples::io::OrganismWriter;
use anyhow::Result;
use std::path::Path;

pub struct Extractor<'a> {
    introns_csv: &'a Path,
    target_dir: &'a Path,
    placeholder_name: &'a str,
}

impl<'a> Extractor<'a> {
    /// # Arguments
    ///
    /// * `placeholder_name` - the source CSV doesn't contain organisms names,
    ///   id est it is *not* known which DNA sample belongs to which organism.
    ///   A placeholder organism name is used instead.
    pub fn new(introns_csv: &'a Path, target_dir: &'a Path, placeholder_name: &'a str) -> Self {
        Self {
            introns_csv,
            target_dir,
            placeholder_name,
        }
    }

    pub fn extract(&self) -> Result<()> {
        let mut reader = CandidateIntronCsvReader::from_file(self.introns_csv)?;
        let mut writer = OrganismWriter::with_target_dir(self.target_dir, self.placeholder_name)?;

        for candidate_intron in reader.introns() {
            let candidate_intron = candidate_intron?;
            writer.write(
                candidate_intron.sequence(),
                candidate_intron.is_positive(),
                "intron",
                0,
                candidate_intron.start(),
                None,
                candidate_intron.scaffold(),
            )?;
        }

        Ok(())
    }
}
