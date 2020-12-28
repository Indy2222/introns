use std::fmt;

#[derive(Clone, Copy)]
pub enum SpliceSiteType {
    Donor,
    Acceptor,
}

impl fmt::Display for SpliceSiteType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Donor => write!(f, "donor"),
            Self::Acceptor => write!(f, "acceptor"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct CompleteType {
    splice_site_type: SpliceSiteType,
    is_positive: bool,
}

impl CompleteType {
    pub fn new(splice_site_type: SpliceSiteType, is_positive: bool) -> Self {
        Self {
            splice_site_type,
            is_positive,
        }
    }

    pub fn is_positive(self) -> bool {
        self.is_positive
    }

    pub fn splice_site_type(self) -> SpliceSiteType {
        self.splice_site_type
    }
}

impl fmt::Display for CompleteType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let positive_str = if self.is_positive() {
            "positive"
        } else {
            "negative"
        };
        write!(f, "{}_{}", positive_str, self.splice_site_type)
    }
}
