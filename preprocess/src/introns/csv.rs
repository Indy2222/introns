use crate::introns::CandidateIntron;
use anyhow::{Context, Result};
use ncrs::data::Symbol;
use std::fs;
use std::io;
use std::path::Path;

pub struct CandidateIntronCsvReader<R: io::Read> {
    inner: csv::Reader<R>,
}

impl CandidateIntronCsvReader<fs::File> {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = fs::File::open(path.as_ref())
            .with_context(|| format!("Error while opening file {}", path.as_ref().display()))?;

        Ok(CandidateIntronCsvReader::new(file))
    }
}

impl<R: io::Read> CandidateIntronCsvReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            inner: csv::ReaderBuilder::new()
                .has_headers(false)
                .delimiter(b';')
                .from_reader(reader),
        }
    }

    pub fn introns(&mut self) -> CandidateIntronIter<'_, R> {
        CandidateIntronIter::new(self.inner.records())
    }
}

pub struct CandidateIntronIter<'a, R: io::Read> {
    inner: csv::StringRecordsIter<'a, R>,
    skipped_headers: bool,
}

impl<'a, R: io::Read> CandidateIntronIter<'a, R> {
    fn new(string_record_iter: csv::StringRecordsIter<'a, R>) -> Self {
        Self {
            inner: string_record_iter,
            skipped_headers: false,
        }
    }

    fn parse_row(
        scaffold: &str,
        start: &str,
        sequence: &str,
        label: &str,
    ) -> Result<CandidateIntron> {
        let scaffold = scaffold.to_owned();
        let start = start
            .parse()
            .context("Couldn't parse intron start position")?;

        let sequence = sequence
            .chars()
            .map(|c| match c {
                'A' | 'a' => Ok(Symbol::Adenine),
                'C' | 'c' => Ok(Symbol::Cytosine),
                'T' | 't' => Ok(Symbol::Thymine),
                'G' | 'g' => Ok(Symbol::Guanine),
                'N' | 'n' => Ok(Symbol::Other),
                _ => bail!("Encountered invalid symbol {}.", c),
            })
            .collect::<Result<Vec<Symbol>>>()?;

        let label = match label.trim() {
            "1" => true,
            "-1" => false,
            _ => bail!("Unexpected label: {}", label),
        };

        Ok(CandidateIntron::new(scaffold, start, sequence, label))
    }
}

impl<'a, R: io::Read> Iterator for CandidateIntronIter<'a, R> {
    type Item = Result<CandidateIntron>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let string_record = match self.inner.next() {
                Some(res) => {
                    let res = res.context("Failed to read line with candidate intron");
                    match res {
                        Ok(v) => v,
                        Err(err) => return Some(Err(err)),
                    }
                }
                None => return None,
            };

            if string_record.len() != 5 {
                return Some(Err(anyhow!(
                    "Invalid number of columns, expected 5 got {}",
                    string_record.len()
                )));
            }

            let scaffold = string_record.get(0).unwrap();
            let start = string_record.get(1).unwrap(); // end is unused
            let sequence = string_record.get(3).unwrap();
            let label = string_record.get(4).unwrap();

            if self.skipped_headers {
                return Some(Self::parse_row(scaffold, start, sequence, label));
            } else {
                if scaffold != "scaffold" {
                    return Some(Err(anyhow!(
                        "Expected columns `scaffold` at positions 0, got: `{}`.",
                        scaffold
                    )));
                }
                if start != "start" {
                    return Some(Err(anyhow!(
                        "Expected columns `start` at positions 1, got: `{}`.",
                        start
                    )));
                }
                if sequence != "sequence" {
                    return Some(Err(anyhow!(
                        "Expected columns `sequence` at positions 3, got: `{}`.",
                        sequence
                    )));
                }
                if label != "label" {
                    return Some(Err(anyhow!(
                        "Expected columns `label` at positions 4, got: `{}`.",
                        label
                    )));
                }

                self.skipped_headers = true;
            }
        }
    }
}
