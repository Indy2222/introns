use crate::features::{FeatureId, SequenceFeature};
use anyhow::{Context, Error, Result};
use std::convert::TryFrom;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub struct Reader<R: io::Read> {
    inner: csv::Reader<R>,
}

impl Reader<fs::File> {
    pub fn from_dir<P: AsRef<Path>>(
        path: P,
        organism_id: &str,
        feature_name: &str,
    ) -> Result<Self> {
        Self::from_file(csv_file_path(path, organism_id, feature_name))
    }

    /// Create a CSV feature reader reading from a file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = fs::File::open(path.as_ref())
            .with_context(|| format!("Error while opening file {}", path.as_ref().display()))?;

        Ok(Reader::new(file))
    }
}

impl<R: io::Read> Reader<R> {
    /// Create a new GFF reader given an instance of `io::Read`, in given format.
    pub fn new(reader: R) -> Self {
        Reader {
            inner: csv::ReaderBuilder::new().from_reader(reader),
        }
    }

    pub fn features(&mut self) -> SequenceFeatures<'_, R> {
        SequenceFeatures {
            inner: self.inner.deserialize(),
        }
    }
}

type SequenceFeatureInner = (String, String, String, u64, u64);

/// An iterator over CSV features loaded from a reader.
pub struct SequenceFeatures<'a, R: io::Read> {
    inner: csv::DeserializeRecordsIter<'a, R, SequenceFeatureInner>,
}

impl<'a, R: io::Read> Iterator for SequenceFeatures<'a, R> {
    type Item = Result<SequenceFeature>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|res| match res {
            Err(e) => Err(Error::new(e)),
            Ok((id, parent_id, scaffold, start, end)) => {
                let id = FeatureId::from_hex(&id)?;
                let parent_id = if parent_id.is_empty() {
                    None
                } else {
                    Some(FeatureId::from_hex(&parent_id)?)
                };
                let start = usize::try_from(start).context("Failed to parse start position")?;
                let end = usize::try_from(end).context("Failed to parse start position")?;

                Ok(SequenceFeature::new(id, parent_id, scaffold, start, end))
            }
        })
    }
}

pub struct Writer<W: io::Write> {
    inner: csv::Writer<W>,
    headers_written: bool,
}

impl Writer<fs::File> {
    /// Write to a given file path.
    pub fn write_to_dir<P: AsRef<Path>>(
        path: P,
        organism_id: &str,
        feature_name: &str,
    ) -> io::Result<Self> {
        Self::write_to_file(csv_file_path(path, organism_id, feature_name))
    }

    /// Write to a given file path.
    pub fn write_to_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        fs::File::create(path).map(Writer::new)
    }
}

impl<W: io::Write> Writer<W> {
    /// Write to a given writer.
    pub fn new(writer: W) -> Self {
        Writer {
            inner: csv::WriterBuilder::new().from_writer(writer),
            headers_written: false,
        }
    }

    /// Write a given GFF record.
    pub fn write(&mut self, record: &SequenceFeature) -> Result<()> {
        if !self.headers_written {
            self.inner
                .serialize(("id", "parent id", "scaffold name", "start", "end"))?;
            self.headers_written = true;
        }

        let parent_id = match record.parent_id() {
            Some(parent_id) => parent_id.to_hex(),
            None => String::new(),
        };
        let start = u64::try_from(record.start).context("Cannot serialize start position")?;
        let end = u64::try_from(record.end).context("Cannot serialize end position")?;
        self.inner
            .serialize((
                record.id().to_hex(),
                parent_id,
                record.scaffold(),
                start,
                end,
            ))
            .map_err(Error::new)
    }
}

fn csv_file_path<P: AsRef<Path>>(dir_path: P, organism_id: &str, feature_name: &str) -> PathBuf {
    let mut path = dir_path.as_ref().to_path_buf();
    path.push(format!("{}_{}.csv", feature_name, organism_id));
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV_FILE: &'static [u8] = b"id,scaffold name,start,end\n\
                                      00000000000000000000000000000001,scaffoldA,10,20\n\
                                      00000000000000000000000000000002,scaffoldB,2,12\n";

    const CSV_FILE_INVALID: &'static [u8] = b"id,scaffold name,start,end\n\
                                              00000000000000000000000000000003,scaffoldA,999999999999999999999999999999999999999910,20\n\
                                              00000000000000000000000000000004,scaffoldB,2,12\n";

    #[test]
    fn test_reader() {
        let mut reader = Reader::new(CSV_FILE);
        let records: Result<Vec<SequenceFeature>> = reader.features().collect();
        let records = records.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].id(),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,]
        );
        assert_eq!(records[0].scaffold(), "scaffoldA");
        assert_eq!(records[0].start(), 10);
        assert_eq!(records[0].end(), 20);
        assert_eq!(
            records[1].id(),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,]
        );
        assert_eq!(records[1].scaffold(), "scaffoldB");
        assert_eq!(records[1].start(), 2);
        assert_eq!(records[1].end(), 12);

        let mut reader = Reader::new(CSV_FILE_INVALID);
        let records: Result<Vec<SequenceFeature>> = reader.features().collect();
        match records {
            Ok(_) => panic!("Reader did not fail."),
            Err(e) => assert_eq!(
                format!("{}", e),
                "CSV deserialize error: record 1 (line: 2, byte: 27): \
                 field 2: number too large to fit in target type"
            ),
        }
    }

    #[test]
    fn test_writer() {
        let mut writer = Writer::new(vec![]);
        writer
            .write(&SequenceFeature::new(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                "scaffoldA".to_owned(),
                10,
                20,
            ))
            .ok()
            .expect("Error writing record");
        writer
            .write(&SequenceFeature::new(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                "scaffoldB".to_owned(),
                2,
                12,
            ))
            .ok()
            .expect("Error writing record");
        assert_eq!(writer.inner.into_inner().unwrap(), CSV_FILE);
    }
}
