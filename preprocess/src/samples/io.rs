use crate::features::FeatureId;
use anyhow::{Context, Error, Result};
use ncrs::data::Symbol;
use ndarray::{arr0, Array, Array2};
use ndarray_npy::NpzWriter;
use std::convert::TryFrom;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

pub struct OrganismWriter {
    target_dir: PathBuf,
    counter: u64,
    path_writer: csv::Writer<File>,
}

impl OrganismWriter {
    pub fn with_target_dir<P: AsRef<Path>>(target_dir: P, organism_id: &str) -> Result<Self> {
        let mut target_dir = target_dir.as_ref().to_path_buf();
        target_dir.push(organism_id);

        if !target_dir.exists() {
            fs::create_dir(&target_dir).context(format!(
                "Failed to create directory for samples of organism {}",
                organism_id
            ))?;
        }

        let path_writer = {
            let mut csv_path = target_dir.clone();
            csv_path.push("samples.csv");

            let create_file = !csv_path.exists();
            let file = OpenOptions::new()
                .create(create_file)
                .append(true)
                .open(&csv_path)
                .with_context(|| {
                    format!("Failed to create samples CSV file {}", csv_path.display())
                })?;

            let mut path_writer = csv::WriterBuilder::new().from_writer(file);

            if create_file {
                path_writer.serialize((
                    "path",
                    "is_positive",
                    "sample_type_name",
                    "feature_id",
                    "absolute_position",
                    "scaffold_name",
                ))?;
            }

            path_writer
        };

        Ok(Self {
            target_dir,
            path_writer,
            counter: 0,
        })
    }

    /// Store a new sample to disk.
    ///
    /// # Arguments
    ///
    /// * `sequence` - input sequence to be stored in the sample.
    ///
    /// * `relative_position` - 0-based position of splice-site within the sequence.
    pub fn write(
        &mut self,
        sequence: &[Symbol],
        is_positive: bool,
        sample_type_name: &str,
        relative_position: usize,
        absolute_position: usize,
        feature_id: Option<FeatureId>,
        scaffold_name: &str,
    ) -> Result<()> {
        let relative_sample_path = self.relative_sample_path(is_positive, sample_type_name);
        let mut target_file_path = self.target_dir.clone();
        target_file_path.push(&relative_sample_path);

        {
            let target_sub_dir = target_file_path.parent().unwrap();
            fs::create_dir_all(&target_sub_dir).with_context(|| {
                format!(
                    "Failed to create samples directory {}",
                    target_sub_dir.display()
                )
            })?;
        }

        let input = sequence_to_one_hot(&sequence);
        let output = arr0(if is_positive { 1.0 } else { 0.0 });

        {
            let file = File::create(&target_file_path).context("Failed to store a sample")?;
            let mut npz_writer = NpzWriter::new_compressed(file);

            let error_context = || {
                format!(
                    "Error while writing to target file {}",
                    target_file_path.display()
                )
            };

            if relative_position > sequence.len() {
                panic!("Relative splice-site position out of sequence bounds.");
            }
            let relative_position = arr0(
                u64::try_from(relative_position)
                    .context("Splice-site position cannot be serialized")?,
            );

            npz_writer
                .add_array("position", &relative_position)
                .with_context(error_context)?;

            npz_writer
                .add_array("input", &input)
                .with_context(error_context)?;
            npz_writer
                .add_array("output", &output)
                .with_context(error_context)?;
        }

        self.write_path(
            &relative_sample_path,
            is_positive,
            sample_type_name,
            feature_id,
            absolute_position,
            scaffold_name,
        )?;

        self.counter += 1;
        Ok(())
    }

    fn write_path(
        &mut self,
        relative_path: &Path,
        is_positive: bool,
        sample_type_name: &str,
        feature_id: Option<FeatureId>,
        absolute_position: usize,
        scaffold_name: &str,
    ) -> Result<()> {
        if relative_path.is_absolute() {
            panic!(
                "Relative sample path expected, got absolute path: {}",
                relative_path.display()
            );
        }

        let feature_id = match feature_id {
            Some(feature_id) => feature_id.to_hex(),
            None => String::new(),
        };

        let absolute_position = format!("{}", absolute_position);

        self.path_writer
            .serialize((
                relative_path
                    .to_str()
                    .expect("Failed to serialize sample path as a UTF-8 string."),
                format!("{}", is_positive),
                sample_type_name,
                feature_id,
                absolute_position,
                scaffold_name,
            ))
            .map_err(Error::new)
    }

    fn relative_sample_path(&self, is_positive: bool, sample_type_name: &str) -> PathBuf {
        let mut relative_path = PathBuf::new();

        relative_path.push(format!("{:04}", self.counter >> 16));
        relative_path.push(format!("{:03}", (self.counter >> 8) & 0xff));

        let mut file_name = String::new();
        if is_positive {
            file_name.push_str("positive_");
        } else {
            file_name.push_str("negative_");
        }
        file_name.push_str(sample_type_name);
        file_name.push_str(&format!("_{:03}.npz", self.counter & 0xff));

        relative_path.push(file_name);

        relative_path
    }
}

/// Create a one-hot-encoded 2D array of a sequence. First dimension
/// represents individual sequence positions and second dimension represents
/// one-hot-encoding.
fn sequence_to_one_hot(sequence: &[Symbol]) -> Array2<f32> {
    let mut input = Array::zeros((sequence.len(), 5));

    for (i, symbol) in sequence.iter().enumerate() {
        let pos: usize = (*symbol).into();
        input[(i, pos)] = 1.;
    }

    input
}
