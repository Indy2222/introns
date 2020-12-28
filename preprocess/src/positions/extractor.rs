use crate::extractor::ParallelExtractor;
use crate::features::{self, FeatureId, SequenceFeature};
use crate::organisms::Organism;
use crate::positions;
use crate::types::{CompleteType, SpliceSiteType};
use crate::utils::load_scaffolds;
use anyhow::{Context, Result};
use ncrs::data::Symbol;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

pub struct Extractor<'a> {
    organism_csv: &'a Path,
    fasta_dir: &'a Path,
    features_dir: &'a Path,
    target_dir: &'a Path,
}

impl<'a> Extractor<'a> {
    pub fn new(
        organism_csv: &'a Path,
        fasta_dir: &'a Path,
        features_dir: &'a Path,
        target_dir: &'a Path,
    ) -> Self {
        Self {
            organism_csv,
            fasta_dir,
            features_dir,
            target_dir,
        }
    }

    fn load_sorted_features(
        &self,
        organism: &Organism,
        feature_name: &str,
    ) -> Result<Vec<SequenceFeature>> {
        let mut feature_reader = features::io::Reader::from_dir(
            &self.features_dir,
            organism.organism_id(),
            feature_name,
        )
        .context("Cannot open gene reader")?;

        let mut features = feature_reader
            .features()
            .collect::<Result<Vec<SequenceFeature>>>()
            .context(format!(
                "Failed to load {} of organism {}",
                feature_name,
                organism.organism_id()
            ))?;

        features.sort_by(|a, b| {
            let str_cmp = a.scaffold().cmp(b.scaffold());
            if str_cmp != Ordering::Equal {
                return str_cmp;
            }

            if a.start() < b.start() {
                Ordering::Less
            } else if a.start() > b.start() {
                Ordering::Greater
            } else if a.end() < b.end() {
                Ordering::Less
            } else if a.end() > b.end() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });

        Ok(features)
    }

    fn extract_splice_sites(&self, organism: &Organism) -> Result<SpliceSites> {
        let mut introns_source = self.load_sorted_features(organism, "introns")?;
        let mut introns_map = HashMap::new();
        let mut introns_tmp = Vec::new();
        let mut last_scaffold = String::new();

        for intron in introns_source.drain(..) {
            if last_scaffold.as_str() != intron.scaffold() {
                if !introns_tmp.is_empty() {
                    introns_map.insert(last_scaffold, introns_tmp);
                    introns_tmp = Vec::new();
                }

                last_scaffold = intron.scaffold().to_owned();
            }
            introns_tmp.push(intron);
        }

        if !introns_tmp.is_empty() {
            introns_map.insert(last_scaffold, introns_tmp);
        }

        Ok(SpliceSites::new(introns_map))
    }
}

impl<'a> ParallelExtractor for Extractor<'a> {
    fn organism_csv(&self) -> &Path {
        self.organism_csv
    }

    fn extract_organism(&self, organism: &Organism) -> Result<()> {
        let genes = self.load_sorted_features(organism, "genes")?;
        let mut splice_sites = self.extract_splice_sites(organism)?;
        let scaffolds = load_scaffolds(self.fasta_dir, organism)?;

        let mut false_donor_buffer = PositionBuffer::new(
            self.target_dir,
            organism.organism_id(),
            CompleteType::new(SpliceSiteType::Donor, false),
        )?;
        let mut false_acceptor_buffer = PositionBuffer::new(
            self.target_dir,
            organism.organism_id(),
            CompleteType::new(SpliceSiteType::Acceptor, false),
        )?;
        let mut true_donor_buffer = PositionBuffer::new(
            self.target_dir,
            organism.organism_id(),
            CompleteType::new(SpliceSiteType::Donor, true),
        )?;
        let mut true_acceptor_buffer = PositionBuffer::new(
            self.target_dir,
            organism.organism_id(),
            CompleteType::new(SpliceSiteType::Acceptor, true),
        )?;

        let mut last_scaffold = String::new();
        let mut last_gene_end = 0;

        for gene in genes {
            if last_scaffold.as_str() != gene.scaffold() {
                last_gene_end = 0;
                last_scaffold = gene.scaffold().to_owned();
            }
            if last_gene_end >= gene.end() {
                continue;
            }
            let gene_start = gene.start().max(last_gene_end);
            last_gene_end = gene.end();

            false_donor_buffer.conditional_flush(gene.scaffold())?;
            false_acceptor_buffer.conditional_flush(gene.scaffold())?;
            true_donor_buffer.conditional_flush(gene.scaffold())?;
            true_acceptor_buffer.conditional_flush(gene.scaffold())?;

            let sequence = match scaffolds.get(gene.scaffold()) {
                Some(scaffold) => scaffold.sequence(),
                None => bail!("Scaffold {} not found", gene.scaffold()),
            };

            splice_sites.advance(gene.scaffold(), gene_start);

            let mut range_start = gene_start;
            if let Some(positions) = splice_sites.donors(gene_start, gene.end()) {
                for (position, intron_id) in positions {
                    find_candidate_donors(
                        sequence,
                        &mut false_donor_buffer,
                        range_start,
                        position,
                        gene.id(),
                    );
                    range_start = position + 1;

                    // utilize find_candidate_donors() for its additional checks
                    // (canonical di-nucleotide fits and the sequence is long
                    // enough)
                    find_candidate_donors(
                        sequence,
                        &mut true_donor_buffer,
                        position,
                        position + 1,
                        intron_id,
                    );
                }
            }
            find_candidate_donors(
                sequence,
                &mut false_donor_buffer,
                range_start,
                gene.end(),
                gene.id(),
            );

            range_start = gene_start;
            if let Some(positions) = splice_sites.acceptors(gene_start, gene.end()) {
                for (position, intron_id) in positions {
                    find_candidate_acceptors(
                        sequence,
                        &mut false_acceptor_buffer,
                        range_start,
                        position,
                        gene.id(),
                    );
                    range_start = position + 1;

                    // utilize find_candidate_acceptors() for its additional checks
                    // (canonical di-nucleotide fits and the sequence is long
                    // enough)
                    find_candidate_acceptors(
                        sequence,
                        &mut true_acceptor_buffer,
                        position,
                        position + 1,
                        intron_id,
                    );
                }
            }
            find_candidate_acceptors(
                sequence,
                &mut false_acceptor_buffer,
                range_start,
                gene.end(),
                gene.id(),
            );
        }

        false_donor_buffer.flush()?;
        false_acceptor_buffer.flush()?;
        true_donor_buffer.flush()?;
        true_donor_buffer.flush()?;

        Ok(())
    }
}

struct PositionBuffer<W: io::Write + io::Seek> {
    writer: positions::io::Writer<W>,
    position_buffer: Vec<usize>,
    feature_id_buffer: Vec<FeatureId>,
    current_scaffold: Option<String>,
}

impl<'a> PositionBuffer<File> {
    fn new(target_dir: &Path, organism_id: &str, complete_type: CompleteType) -> Result<Self> {
        let writer = positions::io::Writer::write_to_dir(target_dir, organism_id, complete_type)?;

        Ok(Self {
            writer,
            position_buffer: Vec::new(),
            feature_id_buffer: Vec::new(),
            current_scaffold: None,
        })
    }
}

impl<W: io::Write + io::Seek> PositionBuffer<W> {
    fn push(&mut self, position: usize, feature_id: FeatureId) {
        self.position_buffer.push(position);
        self.feature_id_buffer.push(feature_id);
    }

    /// Write positions if given scaffold is not current scaffold.
    fn conditional_flush(&mut self, scaffold: &str) -> Result<()> {
        if self.current_scaffold.is_some() && self.current_scaffold.as_ref().unwrap() != scaffold {
            self.flush()?;
        }

        self.current_scaffold = Some(scaffold.to_owned());
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.current_scaffold.is_none() {
            if !self.position_buffer.is_empty() {
                panic!("Current scaffold is not set but buffer is not empty.");
            }

            return Ok(());
        }

        if self.position_buffer.is_empty() {
            return Ok(());
        }

        self.writer.write_scaffold(
            self.current_scaffold.as_ref().unwrap(),
            &self.position_buffer[..],
            &self.feature_id_buffer[..],
        )?;
        self.position_buffer.clear();
        self.feature_id_buffer.clear();
        Ok(())
    }
}

fn find_candidate_donors<W: io::Write + io::Seek>(
    sequence: &[Symbol],
    buffer: &mut PositionBuffer<W>,
    start: usize,
    end: usize,
    feature_id: FeatureId,
) {
    let end = end.min(sequence.len() - 1);
    for i in start..end {
        if sequence[i] == Symbol::Guanine && sequence[i + 1] == Symbol::Thymine {
            buffer.push(i, feature_id);
        }
    }
}

fn find_candidate_acceptors<W: io::Write + io::Seek>(
    sequence: &[Symbol],
    buffer: &mut PositionBuffer<W>,
    start: usize,
    end: usize,
    feature_id: FeatureId,
) {
    let start = start.max(2);
    for i in start..end {
        if sequence[i - 2] == Symbol::Adenine && sequence[i - 1] == Symbol::Guanine {
            buffer.push(i, feature_id);
        }
    }
}

struct SpliceSites {
    introns: HashMap<String, Vec<SequenceFeature>>,
    last_scaffold: String,
    index: usize,
}

impl SpliceSites {
    fn new(introns: HashMap<String, Vec<SequenceFeature>>) -> Self {
        Self {
            introns,
            last_scaffold: String::new(),
            index: 0,
        }
    }

    fn advance(&mut self, scaffold: &str, start: usize) {
        if self.last_scaffold.as_str() != scaffold {
            self.last_scaffold = scaffold.to_owned();
            self.index = 0;
        };

        if let Some(introns) = self.introns.get(scaffold) {
            while self.index < introns.len() && introns[self.index].end() < start {
                self.index += 1;
            }
        }
    }

    fn donors(&self, start: usize, end: usize) -> Option<SpliceSiteIterator> {
        self.positions(StartEnd::Start, start, end)
    }

    fn acceptors(&self, start: usize, end: usize) -> Option<SpliceSiteIterator> {
        self.positions(StartEnd::End, start, end)
    }

    fn positions(
        &self,
        start_end: StartEnd,
        start: usize,
        end: usize,
    ) -> Option<SpliceSiteIterator> {
        match self.introns.get(&self.last_scaffold) {
            Some(introns) => Some(SpliceSiteIterator {
                start_end,
                start,
                end,
                introns: introns.as_slice(),
                index: self.index,
            }),
            None => None,
        }
    }
}

enum StartEnd {
    Start,
    End,
}

struct SpliceSiteIterator<'a> {
    start_end: StartEnd,
    introns: &'a [SequenceFeature],
    index: usize,
    start: usize,
    end: usize,
}

impl<'a> Iterator for SpliceSiteIterator<'a> {
    type Item = (usize, FeatureId);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.index >= self.introns.len() {
                return None;
            }
            let intron = &self.introns[self.index];
            self.index += 1;

            let position = match self.start_end {
                StartEnd::Start => intron.start(),
                StartEnd::End => intron.end(),
            };

            if position < self.start {
                continue;
            } else if position >= self.end {
                return None;
            }

            return Some((position, intron.id()));
        }
    }
}
