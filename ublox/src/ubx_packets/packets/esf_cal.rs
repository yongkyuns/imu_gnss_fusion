#[allow(unused_imports, reason = "It is only unused in some feature sets")]
use crate::FieldIter;
#[cfg(feature = "serde")]
use {super::SerializeUbxPacketFields, crate::serde::ser::SerializeMap};

use crate::{error::ParserError, UbxPacketMeta};
use ublox_derive::ubx_packet_recv;

/// External Sensor Fusion calibrated measurements.
#[ubx_packet_recv]
#[ubx(class = 0x10, id = 0x04, max_payload_len = 1240)]
struct EsfCal {
    s_ttag: u32,
    version: u8,
    reserved0: [u8; 3],
    reserved1: [u8; 4],
    #[ubx(
        map_type = EsfCalDataIter,
        from = EsfCalDataIter::new,
        is_valid = EsfCalDataIter::is_valid,
        may_fail,
    )]
    data: [u8; 0],
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct EsfCalData {
    pub data_type: u8,
    pub data_field: u32,
}

#[derive(Debug, Clone)]
pub struct EsfCalDataIter<'a>(core::slice::ChunksExact<'a, u8>);

impl<'a> EsfCalDataIter<'a> {
    const BLOCK_SIZE: usize = 4;
    fn new(bytes: &'a [u8]) -> Self {
        Self(bytes.chunks_exact(Self::BLOCK_SIZE))
    }

    fn is_valid(bytes: &'a [u8]) -> bool {
        bytes.len() % Self::BLOCK_SIZE == 0
    }
}

impl core::iter::Iterator for EsfCalDataIter<'_> {
    type Item = EsfCalData;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.0.next()?;
        let word = u32::from_le_bytes(chunk.try_into().ok()?);
        Some(EsfCalData {
            data_type: ((word >> 24) & 0x3F) as u8,
            data_field: word & 0x00FF_FFFF,
        })
    }
}
