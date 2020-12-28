use anyhow::{bail, Context, Error, Result};
use ndarray::FixedInitializer;
use sha3::{Digest, Sha3_224};
use std::convert::{TryFrom, TryInto};

pub mod extractor;
pub mod io;

#[derive(Copy, Clone)]
pub struct FeatureId([u8; 16]);

impl FeatureId {
    pub fn from_hex(id: &str) -> Result<Self> {
        if id.len() != 32 {
            bail!(
                "Feature ID has to consist of exactly 32 hexadecimal characters, got: {}",
                id.len()
            );
        }
        let id = hex::decode(id).context("Failed to parse feature ID")?;
        Ok(Self(id[..].try_into().unwrap()))
    }

    pub fn from_bytes(id: [u8; 16]) -> Self {
        Self(id)
    }

    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    pub fn bytes(&self) -> [u8; 16] {
        self.0
    }
}

impl TryFrom<Vec<u8>> for FeatureId {
    type Error = Error;

    fn try_from(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() != 16 {
            bail!("Expected 16 bytes, got: {}.", bytes.len());
        }
        Ok(Self::from_bytes((&bytes[..]).try_into().unwrap()))
    }
}

unsafe impl FixedInitializer for FeatureId {
    type Elem = u8;

    fn as_init_slice(&self) -> &[u8] {
        &self.0
    }

    fn len() -> usize {
        16
    }
}

pub struct SequenceFeature {
    id: FeatureId,
    parent_id: Option<FeatureId>,
    scaffold: String,
    start: usize,
    end: usize,
}

impl SequenceFeature {
    pub fn with_generated_id(
        type_name: &str,
        parent_id: Option<FeatureId>,
        scaffold: String,
        start: usize,
        end: usize,
    ) -> Self {
        let mut hasher = Sha3_224::new();
        hasher.update(type_name.as_bytes());
        if let Some(parent_id) = parent_id {
            hasher.update(parent_id.bytes())
        }
        hasher.update(scaffold.as_bytes());
        hasher.update(u64::try_from(start).unwrap().to_be_bytes());
        hasher.update(u64::try_from(end).unwrap().to_be_bytes());
        hasher.update(scaffold.as_bytes());
        let id = FeatureId::from_bytes(hasher.finalize()[0..16].try_into().unwrap());
        Self::new(id, parent_id, scaffold, start, end)
    }

    pub fn new(
        id: FeatureId,
        parent_id: Option<FeatureId>,
        scaffold: String,
        start: usize,
        end: usize,
    ) -> Self {
        if scaffold.is_empty() {
            panic!("Scaffold has empty name.");
        }

        Self {
            id,
            parent_id,
            scaffold,
            start,
            end,
        }
    }

    pub fn id(&self) -> FeatureId {
        self.id
    }

    pub fn parent_id(&self) -> Option<FeatureId> {
        self.parent_id
    }

    pub fn scaffold(&self) -> &str {
        self.scaffold.as_str()
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }
}
