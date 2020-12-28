use crate::features::FeatureId;
use crate::types::CompleteType;
use anyhow::{Context, Result};
use ndarray::{arr2, Array1, Array2};
use ndarray_npy::{NpzReader, NpzWriter, ReadNpzError};
use std::convert::TryFrom;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use zip::result::ZipError;

pub struct Reader<R: io::Read + io::Seek> {
    inner: NpzReader<R>,
}

impl Reader<fs::File> {
    pub fn from_dir<P: AsRef<Path>>(
        path: P,
        organism_id: &str,
        complete_type: CompleteType,
    ) -> Result<Self> {
        Self::from_file(npz_file_path(path, organism_id, complete_type))
    }

    /// Create a CSV feature reader reading from a file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = fs::File::open(path).context("Failed to open file with positions")?;
        Reader::new(file)
    }
}

impl<R: io::Read + io::Seek> Reader<R> {
    pub fn new(reader: R) -> Result<Self> {
        let inner = NpzReader::new(reader).context("Failed to open NPZ file")?;
        Ok(Reader { inner })
    }

    pub fn read_scaffold_positions(&mut self, scaffold_name: &str) -> Result<Vec<usize>> {
        let res = self.inner.by_name(&format!("positions_{}", scaffold_name));

        // Array is not created for scaffolds with no splice-sites.
        if let Err(ReadNpzError::Zip(ZipError::FileNotFound)) = res {
            return Ok(Vec::new());
        }

        let positions: Array1<u64> =
            res.with_context(|| format!("Failed to read scaffold {}", scaffold_name))?;

        positions
            .into_raw_vec()
            .drain(..)
            .map(|v| usize::try_from(v).context("Failed to parse positions"))
            .collect::<Result<Vec<usize>>>()
    }

    pub fn read_scaffold_feature_ids(&mut self, scaffold_name: &str) -> Result<Vec<FeatureId>> {
        let res = self
            .inner
            .by_name(&format!("feature_ids_{}", scaffold_name));

        // Array is not created for scaffolds with no splice-sites.
        if let Err(ReadNpzError::Zip(ZipError::FileNotFound)) = res {
            return Ok(Vec::new());
        }

        let feature_ids: Array2<u8> =
            res.with_context(|| format!("Failed to read scaffold {}", scaffold_name))?;

        feature_ids
            .outer_iter()
            .map(|v| v.into_owned().into_raw_vec())
            .map(FeatureId::try_from)
            .collect::<Result<Vec<FeatureId>>>()
    }
}

pub struct Writer<W: io::Write + io::Seek> {
    inner: NpzWriter<W>,
}

impl Writer<fs::File> {
    /// Write to a given file path.
    pub fn write_to_dir<P: AsRef<Path>>(
        path: P,
        organism_id: &str,
        complete_type: CompleteType,
    ) -> io::Result<Self> {
        Self::write_to_file(npz_file_path(path, organism_id, complete_type))
    }

    /// Write to a given file path.
    pub fn write_to_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        fs::File::create(path).map(Writer::new)
    }
}

impl<W: io::Write + io::Seek> Writer<W> {
    /// Write to a given writer.
    pub fn new(writer: W) -> Self {
        Writer {
            inner: NpzWriter::new_compressed(writer),
        }
    }

    pub fn write_scaffold(
        &mut self,
        scaffold_name: &str,
        positions: &[usize],
        feature_ids: &[FeatureId],
    ) -> Result<()> {
        if positions.len() != feature_ids.len() {
            panic!("Slice of positions and slice of feature IDs have a different length.");
        }

        let positions = positions
            .iter()
            .map(|v| u64::try_from(*v).context("Failed to serialize a position"))
            .collect::<Result<Vec<u64>>>()?;
        let positions = Array1::from_shape_vec((positions.len(),), positions).unwrap();
        self.inner
            .add_array(format!("positions_{}", scaffold_name), &positions)
            .context("Failed to store positions in a scaffold")?;
        self.inner
            .add_array(format!("feature_ids_{}", scaffold_name), &arr2(feature_ids))
            .context("Failed to store feature IDs in a scaffold")
    }
}

fn npz_file_path<P: AsRef<Path>>(
    dir_path: P,
    organism_id: &str,
    complete_type: CompleteType,
) -> PathBuf {
    let mut path = dir_path.as_ref().to_path_buf();
    path.push(format!("{}_{}.npz", complete_type, organism_id));
    path
}
