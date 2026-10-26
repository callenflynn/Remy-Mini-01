//! Data preparation for the language model: corpus loading, license checks, normalization,
//! deterministic splits, synthetic training and evaluation data (including synthetic tool-use
//! trajectories and their held-out evaluation suite), and chat serialization with
//! assistant-only loss masks. See README.md for the formats and how to add external data.

pub mod chat;
pub mod clean;
pub mod config;
pub mod corpus;
pub mod license;
pub mod loader;
pub mod prepare;
pub mod rng;
pub mod split;
pub mod synthetic;
pub mod tooleval;
pub mod tooluse;
