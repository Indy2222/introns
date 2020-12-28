use ncrs::data::Symbol;

pub mod csv;
pub mod extractor;

pub struct CandidateIntron {
    scaffold: String,
    start: usize,
    sequence: Vec<Symbol>,
    is_positive: bool,
}

impl CandidateIntron {
    pub fn new(scaffold: String, start: usize, sequence: Vec<Symbol>, is_positive: bool) -> Self {
        Self {
            scaffold,
            start,
            sequence,
            is_positive,
        }
    }

    pub fn scaffold(&self) -> &str {
        self.scaffold.as_str()
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn is_positive(&self) -> bool {
        self.is_positive
    }

    pub fn sequence(&self) -> &[Symbol] {
        &self.sequence[..]
    }
}
