//! The DSpark artifact loader — reads the 62-tensor BF16 safetensors, asserts the exact inventory /
//! shapes / dtypes (BF16-only) and an optional sha256 pin, and converts everything to host f32 for
//! the oracle. NO GPU upload in S2 (S3 owns device layout — do not guess it now).

use std::path::Path;

use half::bf16;
use safetensors::SafeTensors;

use crate::dspark::oracle::{DsparkWeights, LayerWeights};
use crate::dspark::{inventory, N_PARAMS, N_TENSORS};

/// A loaded (and validated) artifact + its host-f32 weights.
pub struct LoadedArtifact {
    pub weights: DsparkWeights,
    pub n_tensors: usize,
    pub n_params: u64,
    pub sha256: String,
    pub file_size: u64,
    pub header_size: u64,
}

/// Read `dir/model.safetensors`, validate inventory/shapes/dtypes, optional sha256 pin, and
/// upcast every tensor to f32. Returns an error (not a soft skip) on any mismatch.
pub fn load(dir: &str, sha256_pin: Option<&str>) -> Result<LoadedArtifact, anyhow::Error> {
    let path = Path::new(dir).join("model.safetensors");
    let buf = std::fs::read(&path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let file_size = buf.len() as u64;

    // Optional sha256 pin (full hex; `--probe-dspark-bind`-style reproducibility gate).
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(&buf);
    let hexd: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if let Some(pin) = sha256_pin {
        anyhow::ensure!(
            hexd.eq_ignore_ascii_case(pin),
            "sha256 mismatch: file={hexd} pin={pin}"
        );
    }

    let st = SafeTensors::deserialize(&buf).map_err(|e| anyhow::anyhow!("parse safetensors: {e}"))?;

    // Exact name-set equality vs the 62-tensor inventory (sorted → deterministic comparison).
    let inv = inventory();
    let mut file_names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
    file_names.sort();
    let mut inv_names: Vec<String> = inv.iter().map(|(n, _)| n.clone()).collect();
    inv_names.sort();
    anyhow::ensure!(
        file_names == inv_names,
        "inventory name mismatch (have {} tensors; expected {N_TENSORS})",
        file_names.len()
    );
    anyhow::ensure!(file_names.len() == N_TENSORS, "tensor count {} != {N_TENSORS}", file_names.len());

    // Per-tensor dtype (BF16-only) + exact shape, in FIXED inventory order.
    let mut weights = DsparkWeights {
        layers: Vec::with_capacity(crate::dspark::N_LAYERS),
        fc: Vec::new(),
        hidden_norm: Vec::new(),
        norm: Vec::new(),
        w1: Vec::new(),
        w2: Vec::new(),
        confidence_w: Vec::new(),
        confidence_b: 0.0,
    };

    let bf16_to_f32 = |name: &str, shape: &[usize]| -> Result<Vec<f32>, anyhow::Error> {
        let view = st.tensor(name).map_err(|e| anyhow::anyhow!("tensor {name}: {e}"))?;
        anyhow::ensure!(view.dtype() == safetensors::Dtype::BF16, "{name}: not BF16 ({:?})", view.dtype());
        anyhow::ensure!(view.shape() == shape, "{name}: shape {:?} != expected {shape:?}", view.shape());
        anyhow::ensure!(!name.contains("embed_tokens") && !name.contains("lm_head"),
            "{name}: embed/lm_head must NOT be in the checkpoint (borrowed from the target)");
        let data = view.data();
        Ok(data
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect())
    };

    let mut n_params: u64 = 0;
    for (name, shape) in &inv {
        n_params += shape.iter().map(|&d| d as u64).product::<u64>();
        let v = bf16_to_f32(name, shape)?;
        assign(&mut weights, name, v)?;
    }
    anyhow::ensure!(n_params == N_PARAMS, "param count {n_params} != {N_PARAMS}");

    Ok(LoadedArtifact {
        weights,
        n_tensors: N_TENSORS,
        n_params,
        sha256: hexd,
        file_size,
        header_size: file_size - n_params * 2,
    })
}

/// Assign one f32 tensor into the weight struct by name (fixed-order, deterministic).
fn assign(w: &mut DsparkWeights, name: &str, v: Vec<f32>) -> Result<(), anyhow::Error> {
    match name {
        "fc.weight" => w.fc = v,
        "hidden_norm.weight" => w.hidden_norm = v,
        "norm.weight" => w.norm = v,
        "markov_head.markov_w1.weight" => w.w1 = v,
        "markov_head.markov_w2.weight" => w.w2 = v,
        "confidence_head.proj.weight" => w.confidence_w = v,
        "confidence_head.proj.bias" => w.confidence_b = v[0],
        _ => {
            // layer tensors: `layers.{i}.{suffix}`
            let rest = name.strip_prefix("layers.").ok_or_else(|| anyhow::anyhow!("bad name {name}"))?;
            let (i_str, suffix) = rest.split_once('.').ok_or_else(|| anyhow::anyhow!("bad layer name {name}"))?;
            let i: usize = i_str.parse().map_err(|_| anyhow::anyhow!("bad layer index {name}"))?;
            while w.layers.len() <= i {
                w.layers.push(empty_layer());
            }
            let l = &mut w.layers[i];
            match suffix {
                "self_attn.q_proj.weight" => l.q_proj = v,
                "self_attn.k_proj.weight" => l.k_proj = v,
                "self_attn.v_proj.weight" => l.v_proj = v,
                "self_attn.o_proj.weight" => l.o_proj = v,
                "self_attn.q_norm.weight" => l.q_norm = v,
                "self_attn.k_norm.weight" => l.k_norm = v,
                "input_layernorm.weight" => l.input_ln = v,
                "post_attention_layernorm.weight" => l.post_ln = v,
                "mlp.gate_proj.weight" => l.gate_proj = v,
                "mlp.up_proj.weight" => l.up_proj = v,
                "mlp.down_proj.weight" => l.down_proj = v,
                _ => anyhow::bail!("unknown tensor {name}"),
            }
        }
    }
    Ok(())
}

fn empty_layer() -> LayerWeights {
    LayerWeights {
        q_proj: Vec::new(),
        k_proj: Vec::new(),
        v_proj: Vec::new(),
        o_proj: Vec::new(),
        q_norm: Vec::new(),
        k_norm: Vec::new(),
        input_ln: Vec::new(),
        post_ln: Vec::new(),
        gate_proj: Vec::new(),
        up_proj: Vec::new(),
        down_proj: Vec::new(),
    }
}
