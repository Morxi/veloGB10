use axum::{
    extract::{DefaultBodyLimit, Json, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
    response::sse::Event,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tower_http::cors::{CorsLayer, Any};
use uuid::Uuid;
use chrono;

use crate::batch::{BatchRequest, TokEvent};
use crate::tokenizer::{QwenTokenizer, ChatMessage, ToolCall, ThinkingMode};
use crate::{Usage, Timings, make_timings};

#[derive(Clone)]
pub struct AppState {
    pub scheduler: mpsc::UnboundedSender<BatchRequest>,
    pub tokenizer: Arc<QwenTokenizer>,
    pub model_name: String,
    pub default_max_tokens: usize,
    pub default_rep_penalty: f32,
    pub default_presence_penalty: f32,
    pub default_frequency_penalty: f32,
    /// Server-wide reasoning-effort default from --reasoning-effort. `None` means "unspecified":
    /// the model's OWN chat template picks its baked-in default (Qwen -> `xhigh`, hy_v3 -> `low`),
    /// which is the only value guaranteed to be valid for that family. A request's
    /// `reasoning_effort` field overrides per request.
    pub reasoning_effort: Option<String>,
    /// W1 (Phase 13): the server-wide `--thinking auto|on|off` policy (default Auto = pass nothing,
    /// the model's own chat template decides). A request's `chat_template_kwargs.enable_thinking`
    /// overrides it per call.
    pub thinking: ThinkingMode,
    /// `--output-prompts [cap]`: log every chat-completion request in human-readable form
    /// (effective params, one line per turn, rendered-prompt excerpt up to `cap` chars).
    /// 0 = off (default).
    pub output_prompts: usize,
    /// KV cache depth, in positions. NOTHING used to check a prompt against it: an over-long prompt
    /// ran `write_kv_prefill` straight past the end of the cache and corrupted the next allocation.
    pub max_seq_len: usize,
    /// Scheduler prefix-cache flag (mirror of TpConfig.prefix_cache). The message-boundary
    /// checkpoint (`ckpt_at`) is only ever USED when the scheduler's prefix cache is on
    /// (batch.rs filters it again); gating its render+tokenize here saves the double
    /// template work on every request when the cache is off (TTFT fix (e)).
    pub prefix_cache: bool,
    /// Vision tower (visual trunk) loaded at server start, for image requests. `None` if the
    /// build/model has no vision (text-only server behaves exactly as before).
    pub vision_tower: Option<std::sync::Arc<crate::vision_tower::VisualTower>>,
    /// GPU vision tower (the fast path). When `Some` and `vision_cpu` is false, image requests run
    /// the forward on the GPU; `None` + `vision_cpu: true` (or both unset) keeps the CPU tower.
    pub vision_gpu: Option<std::sync::Arc<std::sync::Mutex<crate::vision_gpu::GpuVisualTower>>>,
    /// Force the CPU vision tower (--vision-cpu), as a diagnostic/escape hatch.
    pub vision_cpu: bool,
    /// Every token id that terminates an assistant turn for this model, resolved once at boot
    /// from the model's own config files (QwenTokenizer::stop_token_ids). Phase-2 A3 uses it to
    /// label a generation's terminal token as a stop token — a fact the Phase-1 ledger could
    /// not recover from the transcripts (harness never reads finish_reason/terminal ids).
    pub stop_ids: Vec<u32>,
    /// OTel generation-telemetry emitter (--otel-endpoint). `None` = OFF (the default): every
    /// telemetry hook site in the SSE path compiles to one `if let Some` branch — zero cost.
    /// Some = the lock-free-ring sink; the SSE chunk hooks below forward the SAME chunk bytes
    /// (single source of truth) and the timer-polled sender exports them (crate::otel).
    pub otel: Option<std::sync::Arc<crate::otel::OtelSink>>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let model = ModelInfo {
        id: state.model_name.clone(),
        object: "model".to_string(),
        created: chrono::Utc::now().timestamp(),
        owned_by: "rust_infer".to_string(),
    };
    Json(ModelList {
        object: "list".to_string(),
        data: vec![model],
    })
}

async fn get_model(State(state): State<AppState>, axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    if id == state.model_name {
        Json(ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "rust_infer".to_string(),
        })
        .into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found. Available: {}", id, state.model_name),
        )
            .into_response()
    }
}

/// The model's PUBLIC id for /v1/models and response `model` fields: the model card's
/// frontmatter `base_model:` line (every model dir ships one, e.g. `base_model: Qwen/Qwen3.8-27B`),
/// falling back to the directory name when the card or the line is absent. Before this, the
/// server reported the lab directory fragment (`"model": "3.8-27b-nvfp4-full-all"`) — an
/// internal path name that no client or catalog can resolve. `--model-name` still overrides.
pub fn model_id_from_dir(model_path: &str) -> String {
    let dir = std::path::Path::new(model_path.trim_end_matches('/'));
    if let Ok(card) = std::fs::read_to_string(dir.join("README.md")) {
        for line in card.lines() {
            let l = line.trim();
            if let Some(v) = l.strip_prefix("base_model:") {
                let v = v.trim().trim_matches('"').trim_matches('\'').trim();
                if !v.is_empty() { return v.to_string(); }
            }
        }
    }
    dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn esc(t: &str) -> String {
    t.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// (The think close marker is resolved per-request from the model's vocab — see
/// QwenTokenizer::think_tags. Qwen: `</think>`; hy_v3: `</think:opensource>`.)

/// Longest suffix of `s` that is a proper (partial) prefix of `marker` — text that could be the start
/// of the marker arriving across decode chunks, and so must be held back rather than forwarded.
/// `--output-prompts [cap]` — log the chat-completion call in human-readable form: effective
/// parameters, one line per message turn, and the exact rendered prompt the model sees
/// (excerpt up to `cap` chars; RUST_INFER_DUMP_PROMPT=1 still writes the full string to /tmp
/// for diffing). Diagnostic output only; nothing here touches the serving path.
#[allow(clippy::too_many_arguments)]
fn log_request_human(
    req: &ChatCompletionRequest,
    effort: Option<&str>,
    prompt: &str,
    prompt_tokens: usize,
    cap: usize,
    model_name: &str,
    render_ms: f64,
) {
    let opt_f32 = |v: &Option<f32>| v.map(|x| x.to_string()).unwrap_or_else(|| "default".into());
    eprintln!("[prompt] ══ chat completion request ({model_name}) ════════════════════════════");
    eprintln!("  stream={}  max_tokens={}  seed={}",
        req.stream,
        req.max_tokens.map(|t| t.to_string()).unwrap_or_else(|| "server-default".into()),
        req.seed.map(|s| s.to_string()).unwrap_or_else(|| "-".into()));
    eprintln!("  temperature={}  top_p={}  top_k={}", req.temperature, req.top_p, req.top_k);
    eprintln!("  penalties: repetition={}  presence={}  frequency={}",
        opt_f32(&req.repetition_penalty), opt_f32(&req.presence_penalty), opt_f32(&req.frequency_penalty));
    eprintln!("  reasoning_effort={} (effective: {})  stop={:?}  include_usage={}",
        req.reasoning_effort.as_deref().unwrap_or("-"),
        effort.unwrap_or("template-default"),
        req.stop,
        req.stream_options.as_ref().map(|s| s.include_usage).unwrap_or(false));
    match &req.tools {
        Some(ts) if !ts.is_empty() => {
            let names: Vec<&str> = ts.iter().filter_map(|t| t.get("function")
                .and_then(|f| f.get("name")).and_then(|n| n.as_str())).collect();
            eprintln!("  tools ({}): {}", ts.len(), names.join(", "));
        }
        _ => eprintln!("  tools: none"),
    }
    eprintln!("  messages ({}):", req.messages.len());
    for (i, m) in req.messages.iter().enumerate() {
        let mut line = format!("    {}. {:9}", i + 1, m.role);
        if let Some(c) = &m.content {
            let flat: String = c.chars().map(|ch| if ch == '\n' { '⏎' } else { ch }).collect();
            let n = flat.chars().count();
            let head: String = flat.chars().take(160).collect();
            line.push_str(&format!(" ({n} ch): {head}{}", if n > 160 { " …" } else { "" }));
        }
        if let Some(tc) = &m.tool_calls {
            let names: Vec<&str> = tc.iter().map(|c| c.function.name.as_str()).collect();
            line.push_str(&format!("  [tool_calls: {}]", names.join(", ")));
        }
        if !m.images.is_empty() { line.push_str(&format!("  [{} image(s)]", m.images.len())); }
        if let Some(id) = &m.tool_call_id { line.push_str(&format!("  [result of {id}]")); }
        eprintln!("{line}");
    }
    let total = prompt.chars().count();
    let trunc = cap.min(total);
    eprintln!("[prompt] rendered prompt: {prompt_tokens} tokens, {total} chars ({render_ms:.1} ms render):");
    let head: String = prompt.chars().take(trunc).collect();
    for l in head.lines() { eprintln!("    | {l}"); }
    if total > trunc {
        eprintln!("    … (+{} more chars of {total} — full dump: RUST_INFER_DUMP_PROMPT=1)", total - trunc);
    }
    eprintln!("[prompt] ══════════════════════════════════════════════════════════════════");
}

fn partial_overlap(s: &str, marker: &str) -> usize {
    (1..marker.len()).rev().find(|&k| s.ends_with(&marker[..k])).unwrap_or(0)
}

fn partial_think_overlap(s: &str, marker: &str) -> usize { partial_overlap(s, marker) }

/// The opening marker of a tool call. While streaming we must never forward this (or a partial prefix
/// of it) to the client as CONTENT: a harness would render raw XML in the chat and never invoke the
/// tool. Once it appears, content emission stops and the rest is buffered for the tool_calls delta.
/// `<tool_call` is the shared PREFIX of qwen's `<tool_call>` and hy_v3's `<tool_call:opensource>` /
/// `<tool_calls:opensource>`, so one constant covers both families.
const TOOL_OPEN: &str = "<tool_call";

/// Split a completed generation into (reasoning, answer). If the close marker is present, everything
/// before it is reasoning (a leading think-open is stripped) and everything after (trimmed) is the
/// answer. If the marker never appears, the whole text is returned as the answer content.
fn split_think(s: &str, think_open: &str, think_close: &str) -> (Option<String>, String) {
    match s.find(think_close) {
        Some(idx) => {
            let mut r = s[..idx].to_string();
            if let Some(rest) = r.strip_prefix(think_open) { r = rest.to_string(); }
            let r = r.trim().to_string();
            let c = s[idx + think_close.len()..].trim_start_matches(['\n', '\r', ' ', '\t']).to_string();
            (if r.is_empty() { None } else { Some(r) }, c)
        }
        None => (None, s.to_string()),
    }
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    /// OpenAI spec requires `model`, but single-model agent clients sometimes omit it. Accept
    /// and fall back to the served model name rather than 422 on a missing field.
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    /// Optional PRNG seed for reproducible sampling (used by stochastic MTP path).
    #[serde(default)]
    seed: Option<u64>,
    /// Stop sequences: accept either a string or a list of strings (OpenAI spec).
    #[serde(default, deserialize_with = "deserialize_stop")]
    stop: Vec<String>,
    /// vLLM-compat: suppress EOS until this many tokens (llama-benchy --exact-tg).
    #[serde(default)]
    min_tokens: Option<usize>,
    /// vLLM-compat: never stop on EOS (--exact-tg).
    #[serde(default)]
    ignore_eos: Option<bool>,
    /// OpenAI tool definitions. Passed straight to the model's chat template, which renders them into
    /// a `# Tools` system block. This field simply did not exist, so serde discarded it and the model
    /// was never told the tools were there -- it answered in prose and every agent harness broke.
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    /// Accepted and echoed for compatibility. We do not force a call: "required"/named choice would
    /// need constrained decoding, and quietly pretending to honour it is worse than not claiming it.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    /// hy_v3 optional reasoning: 'no_think'|'low'|'high', forwarded to the model's chat template.
    /// Per-request override of the server's --reasoning-effort default (which is 'no_think').
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// vLLM's `truncate_prompt_tokens`: keep the LAST n tokens of the prompt (left-truncation)
    /// instead of receiving the over-length 400. Same semantics as /v1/tokenize's field; integer >= 1.
    /// (2026-09-06: this field previously worked on /v1/tokenize ONLY — the chat path silently
    /// dropped it, so the documented escape hatch could not actually rescue an over-length chat.)
    #[serde(default)]
    truncate_prompt_tokens: Option<usize>,
    /// OpenAI streaming options. Only meaningful with stream=true; serde used to drop it silently,
    /// so a client asking for include_usage got nothing and no [DONE] sentinel either.
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    /// Non-standard OpenAI `metadata` object, accepted and passed through. Only one key is
    /// consumed: `session_id` (or `conversation_id`) — the OTel generation-telemetry SESSION
    /// key (see crate::otel::SessionRegistry). Absent/other keys are ignored.
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    /// OpenAI `response_format`. Until Phase 13 this field did not exist on the request struct, so
    /// serde DISCARDED it and every `{"type":"json_schema", ...}` request was silently unconstrained
    /// (the F6 accept-and-ignore class). Now: `json_schema` is compiled into a token-level FSM and
    /// the sampler is masked per step; anything outside the V1 subset is a LOUD 400 naming the
    /// offending keyword. `{"type":"text"}`/absent = unconstrained, exactly as before.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    /// vLLM/SGLang-compatible `chat_template_kwargs`: an ARBITRARY dict forwarded to the model's
    /// own chat template (W1, Phase 13: the customer's `{"enable_thinking": false}` used to be
    /// dropped by serde and the model kept thinking). `enable_thinking` (bool) selects the
    /// template's thinking / no-think branch; every other key passes through verbatim. A
    /// non-object value, or a non-bool `enable_thinking`, is a loud 400 — never accepted-and-ignored.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.8 }
fn default_top_k() -> usize { 20 }

fn deserialize_stop<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    use serde::Deserialize;
    let v: serde_json::Value = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::Null => vec![],
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(a) => a.into_iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => vec![],
    })
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
    /// llama.cpp-compatible timing block, emitted as a top-level extension field (strict
    /// clients ignore unknown top-level fields).
    timings: Timings,
    /// OTel generation-telemetry SESSION key (extension; None when telemetry is off).
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    /// null when the turn is purely a tool call -- that is what OpenAI does, and harnesses key on it.
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ResponseMessage,
    finish_reason: String,
}

/// The scheduler's internal backstop reason (batch.rs) is not an OpenAI value — the spec is
/// stop|length|tool_calls|content_filter. Map it to what it means: generation ran out of room.
fn spec_finish_reason(reason: &str) -> &str {
    if reason == "context_length_exceeded" { "length" } else { reason }
}

/// The chat-template reasoning effort for THIS request: the request's per-call override, else the
/// server's `--reasoning-effort` default, normalized onto THIS model family's template vocabulary.
///
/// SHARED by /v1/chat/completions and the `messages` mode of /v1/tokenize — the two must render the
/// SAME prompt for the same request, or the tokenize-side count diverges from the chat-side
/// `usage.prompt_tokens` (the exact invariant a token-counting client measures).
///
/// Two families, two vocabularies:
///   - hy_v3's template accepts `no_think|low|high` (default low), and its Rust dsv4 path treats
///     None/"" as low.
///   - Qwen3.5's template accepts `xhigh|medium|low` (default xhigh) and RAISES on anything else.
/// Forward the client's value verbatim when it is valid for THIS family's template; convert
/// the OpenAI API convention onto the nearest native level. When NEITHER the request nor
/// --reasoning-effort specifies one, pass None so the model's own template default wins
/// (xhigh for Qwen 3.8, low for hy_v3) — never a hardcoded guess.
///
/// Qwen 3.8 native (the ONLY values its template accepts; anything else raises => 500):
///   xhigh (default) | medium | low          ["no_think"/"off" => enable_thinking=false]
/// OpenAI API -> Qwen 3.8 (owner spec 2026-08-30):
///   none    -> thinking off      (latency-critical; no reasoning)
///   low     -> low               (efficient reasoning)
///   medium  -> medium            (balanced; OpenAI's default)
///   high    -> xhigh             (hard reasoning)
///   xhigh   -> xhigh             (deep research)
///   max     -> xhigh             (maximum)
/// hy_v3 native: no_think | low | high  =>  none->no_think, low->low, medium/high/xhigh/max->high
///
/// REGRESSION FIX (2026-08-30): the 289e1a1 refactor lumped "high" into the no_think arm,
/// so every OpenAI-convention client sending reasoning_effort=high silently LOST thinking.
fn resolve_reasoning_effort(tokenizer: &QwenTokenizer, req_effort: Option<&str>,
                            server_default: Option<&str>) -> Option<String> {
    let (_, think_close_tag, _) = tokenizer.think_tags();
    let hy_family = think_close_tag != "</think>";
    req_effort.or(server_default).map(|e| {
        let n = match (e, hy_family) {
            ("high", true) | ("medium", true) | ("xhigh", true) | ("max", true) => "high",
            ("high", false) | ("xhigh", false) | ("max", false) => "xhigh",
            ("low", _) | ("medium", false) => e,
            ("no_think", _) | ("none", _) | ("minimal", _) | ("off", _) | ("", _) => "no_think",
            (other, _) => other,
        };
        if n != e {
            eprintln!("[req] reasoning_effort '{e}' normalized to '{n}' for this model family");
        }
        n.to_string()
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Log the request parameters the client sent (useful for debugging OpenWebUI behavior)
    eprintln!(
        "[req] params  temp={:?} top_p={:?} top_k={} max_tok={:?} rep_pen={:?} presence={:?} freq={:?} stream={}",
        req.temperature, req.top_p, req.top_k,
        req.max_tokens, req.repetition_penalty, req.presence_penalty, req.frequency_penalty,
        req.stream
    );
    if let Some(t) = &req.tools {
        let names: Vec<&str> = t.iter()
            .filter_map(|x| x.pointer("/function/name").and_then(|v| v.as_str())).collect();
        eprintln!("[req] tools   {} offered: {:?} tool_choice={:?}", t.len(), names, req.tool_choice);
    }
    // Resolve the effective reasoning effort. Two families, two vocabularies:
    //   - hy_v3's template accepts `no_think|low|high` (default low), and its Rust dsv4 path treats
    //     None/"" as low.
    //   - Qwen3.5's template accepts `xhigh|medium|low` (default xhigh) and RAISES on anything else.
    // Forward the client's value verbatim when it is valid for THIS family's template; convert
    // the OpenAI API convention onto the nearest native level. When NEITHER the request nor
    // --reasoning-effort specifies one, pass None so the model's own template default wins
    // (xhigh for Qwen 3.8, low for hy_v3) — never a hardcoded guess.
    //
    // Qwen 3.8 native (the ONLY values its template accepts; anything else raises => 500):
    //   xhigh (default) | medium | low          ["no_think"/"off" => enable_thinking=false]
    // OpenAI API -> Qwen 3.8 (owner spec 2026-08-30):
    //   none    -> thinking off      (latency-critical; no reasoning)
    //   low     -> low               (efficient reasoning)
    //   medium  -> medium            (balanced; OpenAI's default)
    //   high    -> xhigh             (hard reasoning)
    //   xhigh   -> xhigh             (deep research)
    //   max     -> xhigh             (maximum)
    // hy_v3 native: no_think | low | high  =>  none->no_think, low->low, medium/high/xhigh/max->high
    //
    // REGRESSION FIX (2026-08-30): the 289e1a1 refactor lumped "high" into the no_think arm,
    // so every OpenAI-convention client sending reasoning_effort=high silently LOST thinking.
    // W2 (Phase 13): `response_format` — compiled HERE, before any work: an unsupported construct
    // is a 400 naming the keyword (never accepted-and-ignored), a supported schema arms the mask.
    // Phase-13 W2 STATUS (2026-09-13): the schema compiler, the token-level FSM, the mask kernels
    // and the scheduler plumbing are all in this build (unit-tested), but end-to-end ENFORCEMENT
    // could not be verified in the phase's boot budget: the mask is armed and restrictive
    // ("[schema] verify mask armed: ... allowed_tokens_pos0=2") yet a token outside it still
    // reached the stream, so this build REFUSES schema requests loudly instead of accepting them
    // and quietly ignoring the constraint — the F6 class this work exists to delete. Flip
    // JSON_SCHEMA_ENFORCEMENT_ENABLED to true once the emission path is proven end-to-end.
    const JSON_SCHEMA_ENFORCEMENT_ENABLED: bool = false;
    let mut schema_mask: Option<std::sync::Arc<crate::json_schema::SchemaMask>> = None;
    if let Some(rf) = req.response_format.as_ref() {
        match crate::json_schema::compile_response_format(rf) {
            Ok(None) => {}
            Ok(Some(mut m)) if JSON_SCHEMA_ENFORCEMENT_ENABLED => {
                m.set_vocab(state.tokenizer.vocab_pieces());
                eprintln!("[req] response_format: constrained decoding armed ({})", m.summary());
                schema_mask = Some(std::sync::Arc::new(m));
            }
            Ok(Some(m)) => {
                return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
                    "message": format!(
                        "this build does not enforce json_schema constrained decoding yet \
                         (Phase 13 W2 partial): every token would be unconstrained, so the \
                         request is refused instead of accepted-and-ignored. Drop response_format \
                         or use {{\"type\":\"text\"}}. Parsed schema: {}",
                        m.summary()),
                    "type": "invalid_request_error", "code": "json_schema_not_enforced",
                }}))).into_response();
            }
            Err(e) => {
                return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
                    "message": e, "type": "invalid_request_error", "code": "unsupported_response_format",
                }}))).into_response();
            }
        }
    }
    // W1: `chat_template_kwargs` (arbitrary dict → the model's own template). Validate LOUDLY
    // before any render: a malformed value used to be invisible (serde dropped the field), and
    // the customer was left with a model that ignored the request. 400, never accept-and-ignore.
    if let Err(e) = crate::tokenizer::enable_thinking_kwarg(req.chat_template_kwargs.as_ref()) {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": e.to_string(), "type": "invalid_request_error", "code": "invalid_chat_template_kwargs",
        }}))).into_response();
    }
    let effort_owned = resolve_reasoning_effort(&state.tokenizer, req.reasoning_effort.as_deref(),
                                                state.reasoning_effort.as_deref());
    let effort: Option<&str> = effort_owned.as_deref();
    // OpenAI/vLLM tool_choice contract. The engine has no guided-decode forcing, so the
    // semantics are implemented at the prompt level (the same approach llama.cpp's server
    // takes): "none" removes the tools entirely (the model cannot call what it cannot see);
    // "required" / a specific function append an explicit forcing instruction naming the
    // constraint. Scoring harnesses (tool-eval-bench) drive scenarios through these modes.
    let forced_fn: Option<String> = match req.tool_choice.as_ref() {
        Some(v) => {
            let s = v.as_str().unwrap_or("");
            if s == "none" { Some("__none__".to_string()) }
            else if s == "required" || s == "auto" { None }  // auto = no forcing
            else { v.pointer("/function/name").and_then(|n| n.as_str()).map(|n| n.to_string()) }
        }
        None => None,
    };
    let tools_for_template = if forced_fn.as_deref() == Some("__none__") { None } else { req.tools.as_deref() };
    let mut messages = req.messages.clone();
    {
        let force_line: Option<String> = match forced_fn.as_deref() {
            Some("__none__") => None, // tools already removed; nothing to force
            Some(name) => Some(format!("IMPORTANT: You MUST call the function `{name}` with appropriate arguments before answering. Do not answer in plain text.")),
            None if req.tool_choice.as_ref().and_then(|v| v.as_str()) == Some("required") =>
                Some("IMPORTANT: You MUST use one of the provided tools to answer. Do not answer in plain text.".to_string()),
            None => None,
        };
        if let Some(line) = force_line {
            // Append to the LAST message: recency dominates instruction-following; a
            // system-level line is routinely outweighed by the turn's own phrasing.
            if let Some(last) = messages.last_mut() {
                match &mut last.content {
                    Some(c) => { c.push_str("\n\n"); c.push_str(&line); }
                    None => { last.content = Some(line); }
                }
            }
        }
    }
    let t_render = std::time::Instant::now();
    let prompt = match state.tokenizer.apply_chat_template(&messages, tools_for_template, effort,
                                                          req.chat_template_kwargs.as_ref(),
                                                          state.thinking) {
        Ok(p) => p,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let render_ms = t_render.elapsed().as_secs_f64() * 1000.0;

    // Optional diagnostic: dump the exact rendered prompt string so the bytes a model
    // actually sees can be inspected/diffed across models or turns. Enable with
    // RUST_INFER_DUMP_PROMPT=1. Writes /tmp/rust_infer_prompt_<n>.txt per request.
    if std::env::var("RUST_INFER_DUMP_PROMPT").is_ok() {
        static DUMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = DUMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dump_path = format!("/tmp/rust_infer_prompt_{}.txt", n);
        if std::fs::write(&dump_path, &prompt).is_ok() {
            eprintln!("[req] dumped prompt ({} chars) -> {}", prompt.chars().count(), dump_path);
        }
    }

    let t_encode = std::time::Instant::now();
    let mut prompt_tokens = match state.tokenizer.encode(&prompt, true) {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
    // TTFT fix 0 attribution (GB10_PREFILL_TRACE): the request-path pre-model costs.
    if crate::env_knob("GB10_PREFILL_TRACE", "DSV4_PREFILL_TRACE").is_some() {
        eprintln!("[pf] server render={render_ms:.3}ms encode={encode_ms:.3}ms");
    }

    // V3 vision dispatch: if any message carries images (and the server has a vision tower),
    // decode+preprocess+run the tower, expand the image_pad span in the token stream, and carry the
    // merged embeddings + spans for the model prefill splice. Text-only traffic is unchanged.
    let mut image_embeds: Option<Vec<f32>> = None;
    let mut image_spans: Vec<crate::vision_encoder::ImageSpan> = Vec::new();
    let urls: Vec<String> = req.messages.iter()
        .flat_map(|m| m.images.iter().filter_map(|i| i.url.clone()))
        .collect();
    if !urls.is_empty() {
        let vt0 = std::time::Instant::now();
        // Prefer the GPU tower (fast path) unless --vision-cpu forces the CPU reference.
        let prep = if let Some(g) = state.vision_gpu.clone() {
            if !state.vision_cpu {
                let mut gvt = g.lock().expect("vision_gpu lock");
                crate::vision_encoder::prepare_vision_request_gpu(&mut gvt, &urls, &prompt_tokens)
            } else {
                state.vision_tower.as_ref().map(|t| crate::vision_encoder::prepare_vision_request(t, &urls, &prompt_tokens)).unwrap_or_else(|| Err(anyhow::anyhow!("vision_gpu forced but no CPU tower")))
            }
        } else if let Some(tower) = state.vision_tower.clone() {
            crate::vision_encoder::prepare_vision_request(&tower, &urls, &prompt_tokens)
        } else {
            Err(anyhow::anyhow!("no vision tower loaded"))
        };
        eprintln!("[vision] dispatch {} images, prepare took {} ms, len={}",
            urls.len(), vt0.elapsed().as_millis(), prep.as_ref().map(|p| p.image_embeds.len()).unwrap_or(0));
        match prep {
            Ok(prep) => {
                prompt_tokens = prep.expanded_tokens;
                image_embeds = Some(prep.image_embeds);
                image_spans = prep.spans;
            }
            Err(e) => return (StatusCode::BAD_REQUEST,
                format!("vision preprocessing failed: {e}")).into_response(),
        }
    }

    // truncate_prompt_tokens (vLLM's field): keep the LAST n tokens — the same left-truncation
    // convention the /v1/tokenize handler implements. Runs BEFORE the max_seq_len check so any n
    // that fits turns the over-length 400 into a served request; the boundary snapshot (ckpt_at)
    // below re-renders the FULL template, so a truncated stream simply fails its `n < prompt_len`
    // filter and skips the checkpoint this turn (truncated history is not a prefix of the next
    // full-history turn anyway — the cache can't be trusted across the cut).
    if let Some(n) = req.truncate_prompt_tokens {
        if n == 0 {
            return (StatusCode::BAD_REQUEST,
                    "'truncate_prompt_tokens' must be an integer >= 1").into_response();
        }
        if prompt_tokens.len() > n {
            let dropped = prompt_tokens.len() - n;
            prompt_tokens.drain(..dropped); // keep the LAST n — vLLM's left-truncation convention
            eprintln!("[req] truncate_prompt_tokens: dropped the OLDEST {dropped} of {} prompt \
                       tokens (kept the last {n})", dropped + n);
        }
    }

    let prompt_len = prompt_tokens.len();

    if state.output_prompts > 0 {
        let mname = req.model.clone().unwrap_or_else(|| state.model_name.clone());
        log_request_human(&req, effort, &prompt, prompt_len, state.output_prompts, &mname, render_ms);
    }

    // Where to snapshot the GDN state: the message boundary, i.e. this prompt without its trailing
    // generation prompt. Everything up to here is what the NEXT turn replays verbatim. Rendering the
    // template a second time costs microseconds and saves a whole re-prefill per turn — but only
    // when the scheduler's prefix cache is actually on (batch.rs filters ckpt_at again); with the
    // cache off this second render+encode is pure TTFT cost (fix (e), EXPERT_TTFT_PREFILL_RESPONSE).
    let ckpt_at = if state.prefix_cache {
        state.tokenizer
            .apply_chat_template_no_gen(&req.messages, req.tools.as_deref(), effort,
                                        req.chat_template_kwargs.as_ref(), state.thinking).ok()
            .and_then(|s| state.tokenizer.encode(&s, true).ok())
            .map(|t| t.len())
            .filter(|&n| n > 0 && n < prompt_len)
    } else { None };

    // The KV cache holds exactly `max_seq_len` positions. A prompt past that end used to be written
    // out of bounds — silently, corrupting whatever allocation followed, which showed up as two
    // identical prefills disagreeing. Reject what cannot fit, and cap generation at the room left:
    // running short is a `finish_reason: "length"`, which is in the contract. Corruption is not.
    if prompt_len >= state.max_seq_len {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": format!("This model's maximum context length is {} tokens, but your messages \
                                came to {} tokens. Shorten the input or restart the server with a \
                                larger --max-seq-len.", state.max_seq_len, prompt_len),
            "type": "invalid_request_error", "code": "context_length_exceeded",
        }}))).into_response();
    }
    let room = state.max_seq_len - prompt_len;
    let asked = req.max_tokens.unwrap_or(state.default_max_tokens);
    let req_max = asked.min(room);
    // If the KV cache forced generation shorter than asked, SAY SO. A thinking model spends a big fixed
    // chunk on its <think> block, so a silently-shrunk budget looks like "truncated output / only
    // reasoning" as a conversation grows — which is exactly how this surfaced in the wild. Raise
    // --max-seq-len (graphs cost ~nothing here; KV is ~64 KB/token) to give multi-turn room.
    if req_max < asked {
        eprintln!("[req] max_tokens clamped {} -> {} (KV cache room: {} of {} used by the {}-token prompt; \
                   raise --max-seq-len)", asked, req_max, prompt_len, state.max_seq_len, prompt_len);
    }
    let temperature = req.temperature;
    let top_p = req.top_p.max(0.01);

    // Submit to the batching scheduler and receive tokens on a channel.
    // Use request's penalties if explicitly set, else fall back to server defaults.
    let rep_penalty = req.repetition_penalty.unwrap_or(state.default_rep_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(state.default_presence_penalty);
    let pp_source = if req.presence_penalty.is_some() { "request" } else { "server-default" };
    let frequency_penalty = req.frequency_penalty.unwrap_or(state.default_frequency_penalty);

    let (tx, mut rx) = mpsc::unbounded_channel::<TokEvent>();
    let request = BatchRequest {
        prompt: prompt_tokens.clone(),
        max_new: req_max,
        temperature,
        received_at: std::time::Instant::now(),
        top_p,
        top_k: req.top_k,
        rep_penalty,
        presence_penalty,
        frequency_penalty,
        min_new: req.min_tokens.unwrap_or(0),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        tx,
        seed: req.seed,
        ckpt_at,
        domain: crate::batch::classify_domain(&prompt),
        image_embeds,
        image_spans,
        schema: schema_mask.clone(),
    };
    let _ = state.scheduler.send(request);

    // SESSION identity for the OTel generation-telemetry (crate::otel::SessionRegistry). One
    // resolution per REQUEST, only when the emitter is on (off = no session work at all):
    // explicit client key first (X-Session-Id header, then metadata.session_id / .conversation_id),
    // else the engine infers the continuous conversation from the messages-prefix rule — turn
    // N+1's messages array contains turn N's as an exact prefix, so a follow-up lands in the
    // SAME session id and the client sees one continuous session, not one per response.
    let otel_session: Option<String> = state.otel.as_ref().map(|s| {
        let explicit = headers.get("x-session-id").and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .or_else(|| req.metadata.as_ref().and_then(|m| m.get("session_id"))
                .and_then(|v| v.as_str()).map(str::to_string))
            .or_else(|| req.metadata.as_ref().and_then(|m| m.get("conversation_id"))
                .and_then(|v| v.as_str()).map(str::to_string));
        // Canonical per-message JSON, in order — the fingerprint material (serialized once).
        let per_msg: Vec<String> = req.messages.iter()
            .map(|m| serde_json::to_string(m).unwrap_or_default()).collect();
        let sess = s.resolve_session(explicit, &per_msg);
        eprintln!("[req] session {sess} (turn of {} message(s))", req.messages.len());
        sess
    });

    let content_chunk = |cid: &str, created: i64, model: &str, text: &str| {
        format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
            cid, created, model, esc(text))
    };
    let tool_calls_chunk = |cid: &str, created: i64, model: &str, calls: &[ToolCall]| {
        let arr: Vec<serde_json::Value> = calls.iter().enumerate().map(|(i, c)| serde_json::json!({
            "index": i, "id": c.id, "type": c.kind,
            "function": {"name": c.function.name, "arguments": c.function.arguments},
        })).collect();
        serde_json::json!({
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"tool_calls": arr}, "finish_reason": null}],
        }).to_string()
    };
    let reasoning_chunk = |cid: &str, created: i64, model: &str, text: &str| {
        format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"reasoning_content\":\"{}\"}},\"finish_reason\":null}}]}}",
            cid, created, model, esc(text))
    };

    if req.stream {
        eprintln!("[req] stream  prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let tokenizer = Arc::clone(&state.tokenizer);
        let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());
        let stops = req.stop.clone();
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let t0 = std::time::Instant::now();
        let req_tools = req.tools.clone();
        let include_usage = req.stream_options.as_ref().map(|o| o.include_usage).unwrap_or(true);
        // Owned copy: the SSE generator is 'static, so it cannot capture the `&str` that borrows
        // effort_owned (A3 log line only; the render/ckpt calls above still use `effort`).
        let effort_for_log: Option<String> = effort.map(|s| s.to_string());
        // Think markers + the initial reasoning/content state. Derive the start mode from the
        // RENDERED PROMPT TAIL, not a family constant: qwen's template primes an OPEN think block
        // when thinking (prompt ends with `<think>`), but its no-think branch (enable_thinking=
        // false — effort none/no_think/off) emits a CLOSED empty block `<think>\n\n</think>\n\n`,
        // so that stream starts in CONTENT. A family-constant `primed` mislabeled the direct
        // answer as reasoning_content forever (content stayed empty; the model card's non-thinking
        // mode is real and respected by the model — verified sync+bf16 2026-08-30). hy_v3 keeps
        // the effort arm: its low|high template primes `…assistant<think:opensource>` even when
        // the tail check can't see it.
        let (think_open, think_close, _) = tokenizer.think_tags();
        // trim_end: the primed form is `<think>\n` — the trailing newline must not defeat the
        // tail check (a plain ends_with(think_open) sent thinking streams to content: exactly
        // the 35b4b15 follow-up bug). The no-think tail `<think>\n\n</think>\n\n` trims to
        // `</think>` and stays content.
        let starts_in_reasoning = prompt.trim_end().ends_with(think_open)
            || matches!(effort, Some("low") | Some("high"));

        // OTel generation telemetry (--otel-endpoint). Handle built ONCE per request; the hooks
        // below forward the SAME chunk strings the SSE path yields (single source of truth —
        // the telemetry path never re-derives or re-encodes a delta). `None` (endpoint absent)
        // = the default = every hook below compiles away. The hooks are pure observers: they
        // cannot alter the SSE bytes (see crate::otel — lock-free ring, drop-on-full, the
        // sender runs on its own timer off the compute stream). request.id = the SESSION key
        // (stable across the conversation's turns); generation.id = this POST's execution id.
        let otel_req = match (&state.otel, &otel_session) {
            (Some(s), Some(sess)) =>
                Some(s.open_request(sess, &format!("gen-{}", Uuid::new_v4()))),
            _ => None,
        };

        let stream = async_stream::stream! {
            let role_chunk = format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
                completion_id, created, model_name);
            if let Some(r) = &otel_req { r.start(&role_chunk); }   // event=stream_start, token.index 0
            yield Ok::<Event, axum::Error>(Event::default().data(role_chunk));
            // Byte-level stream decoder: the per-token decode path above would mangle every
            // multi-byte char split across tokens (all emoji) into "�" — the crate's ByteLevel
            // decode is String::from_utf8_lossy per call. Reassembles raw bytes across tokens.
            let mut stream_dec = tokenizer.stream_decoder();
            let mut acc = String::new();
            let mut n = 0usize;
            let mut last_tok: Option<u32> = None;
            let mut stop_hit = false;
            let mut finish = "length".to_string();
            let mut first_tok: Option<std::time::Instant> = None;
            // Thinking-model split: qwen's prompt is primed with `<think>\n`, so the generated stream
            // is `…reasoning…</think>\n\nanswer`. Pre-close text -> reasoning_content, post-close
            // -> content. hy_v3's no_think prompt already closed the (empty) block, so it starts as
            // content. The close marker may span decode chunks, so we hold back a tail that could be
            // its prefix until more text arrives.
            let mut content_start: Option<usize> = if starts_in_reasoning { None } else { Some(0) };
            let mut reason_emitted: usize = 0;
            let mut content_emitted: usize = 0;
            while let Some(ev) = rx.recv().await {
                match ev {
                    TokEvent::Tok(t) => {
                        n += 1;
                        last_tok = Some(t);
                        if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                        let text = stream_dec.push(t);
                        if !text.is_empty() {
                            acc.push_str(&text);
                                match content_start {
                                    None => {
                                        // Search the close tag from reason_emitted, not from 0:
                                        // a second think block must not match the first one's close.
                                        if let Some(idx) = acc[reason_emitted..].find(think_close).map(|i| reason_emitted + i) {
                                            if idx > reason_emitted {
                                                let c = reasoning_chunk(&completion_id, created, &model_name, &acc[reason_emitted..idx]);
                                                if let Some(r) = &otel_req { r.delta(&c); }
                                                yield Ok(Event::default().data(c));
                                            }
                                            let cs = idx + think_close.len();
                                            let mut lead = cs;
                                            while lead < acc.len() && matches!(acc.as_bytes()[lead], b'\n' | b'\r' | b' ' | b'\t') { lead += 1; }
                                            content_start = Some(lead);
                                            // Same hold-back as the steady-state content branch
                                            // below: if a tool-call marker arrived in the same
                                            // decode chunk as the think close, it must NOT be
                                            // forwarded as content.
                                            let region = &acc[lead..];
                                            let safe_end = match region.find(TOOL_OPEN) {
                                                Some(i) => lead + i,
                                                None => acc.len() - partial_overlap(region, TOOL_OPEN)
                                                    .max(partial_overlap(region, think_open)),
                                            };
                                            if safe_end > lead {
                                                let c = content_chunk(&completion_id, created, &model_name, &acc[lead..safe_end]);
                                                if let Some(r) = &otel_req { r.delta(&c); }
                                                yield Ok(Event::default().data(c));
                                            }
                                            content_emitted = safe_end;
                                        } else {
                                            let overlap = partial_think_overlap(&acc, think_close);
                                            let safe = (acc.len() - overlap).max(reason_emitted);
                                            // Tool-call hold-back in REASONING mode too. A model
                                            // that calls a tool without ever emitting `</think>`
                                            // (qwen's first-turn behavior on trivial calls: the
                                            // template primes `<think>` and the model jumps
                                            // straight to the call) stays in this branch, which
                                            // had NO TOOL_OPEN hold-back — the raw call markup
                                            // streamed out as reasoning_content while
                                            // finalize_parsed ALSO emitted the structured
                                            // tool_calls delta: the client saw the same call
                                            // twice (2026-08-30 user report). Same contract as
                                            // the content branch below: once TOOL_OPEN appears,
                                            // reasoning emission stops; the buffer is either
                                            // parsed into the tool_calls delta or surfaced
                                            // post-loop by held_back_remainder.
                                            let region = &acc[reason_emitted..safe];
                                            let safe_end = match region.find(TOOL_OPEN) {
                                                Some(i) => reason_emitted + i,
                                                None => safe - partial_overlap(region, TOOL_OPEN),
                                            };
                                            if safe_end > reason_emitted {
                                                let c = reasoning_chunk(&completion_id, created, &model_name, &acc[reason_emitted..safe_end]);
                                                if let Some(r) = &otel_req { r.delta(&c); }
                                                yield Ok(Event::default().data(c));
                                                reason_emitted = safe_end;
                                            }
                                        }
                                    }
                                    Some(cs) => {
                                        // If the model OPENS a think block (hy_v3 with
                                        // reasoning_effort low|high), hand off to the reasoning
                                        // branch: emit the content before the marker, then split
                                        // reasoning until the close tag. Without this the raw
                                        // think tags would leak into `content`.
                                        let region = &acc[cs..];
                                        if let Some(tp) = region.find(think_open) {
                                            let upto = cs + tp;
                                            if upto > content_emitted {
                                                let c = content_chunk(&completion_id, created, &model_name, &acc[content_emitted..upto]);
                                                if let Some(r) = &otel_req { r.delta(&c); }
                                                yield Ok(Event::default().data(c));
                                            }
                                            content_start = None;
                                            reason_emitted = upto + think_open.len();
                                        } else {
                                        // Hold back anything that is, or could become, a tool call.
                                        // Forwarding `<tool_call>` as content makes the harness render
                                        // XML in the chat and never invoke the tool. Same hold-back
                                        // for a think-open prefix spanning decode chunks.
                                        let safe_end = match region.find(TOOL_OPEN) {
                                            Some(i) => cs + i,          // a call has started: emit nothing more
                                            None => acc.len() - partial_overlap(region, TOOL_OPEN)
                                                .max(partial_overlap(region, think_open)),
                                        };
                                        if safe_end > content_emitted {
                                            let c = content_chunk(&completion_id, created, &model_name, &acc[content_emitted..safe_end]);
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
                                            content_emitted = safe_end;
                                        }
                                        }
                                    }
                                }
                            }
                        if !stops.is_empty() {
                            if let Some(p) = stops.iter().filter_map(|s| acc.find(s)).min() {
                                acc.truncate(p);
                                stop_hit = true;
                                finish = "stop".to_string();
                                break;
                            }
                        }
                    }
                    TokEvent::Finish { reason } => { finish = reason; break; }
                }
            }
            // The call was buffered, not streamed (see the hold-back above). The DECISION is
            // crate::tools::finalize_parsed — the one canonical serializer shared with the
            // non-streaming mode — and the held-back text is surfaced by
            // tools::held_back_remainder, so streaming can never silently drop text the JSON
            // mode returns (2026-08-27 user report: a malformed `function=NAME>` block with the
            // `<` missing vanished from the SSE stream while the JSON response leaked it).
            let (_, done_content) = split_think(&acc, think_open, think_close);
            let parsed = crate::tools::parse(&done_content, req_tools.as_deref());
            if req_tools.is_some() {
                let dump = std::env::var("RUST_INFER_DUMP_TOOLS").is_ok();
                if dump || parsed.tool_calls.is_empty() {
                    eprintln!("[req] raw model output ({} chars): {:?}", done_content.chars().count(),
                              done_content.chars().take(1200).collect::<String>());
                }
            }
            // Phase-4 A2: same empty-argument alarm as the non-streaming path — an agent harness
            // streams, so this branch is the one that actually gets used.
            for a in crate::tools::empty_arg_alarms(&done_content, req_tools.as_deref(), &parsed.tool_calls) {
                eprintln!("[tool-args-alarm] {a}");
            }
            let (_, tool_calls, fin) = crate::tools::finalize_parsed(&done_content, parsed, &finish);
            if !tool_calls.is_empty() {
                // Log the ARGUMENTS, not just the names — see the note on the non-streaming path.
                // Agent harnesses stream, so this is the branch that actually gets used, and it was the
                // one printing a bare `tool_calls 1: ["write"]` while a file silently failed to appear.
                for t in &tool_calls {
                    eprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
                }
                let tc = tool_calls_chunk(&completion_id, created, &model_name, &tool_calls);
                if let Some(r) = &otel_req { r.delta(&tc); }
                yield Ok(Event::default().data(tc));
                finish = fin;
            // Watermark is whichever cursor is live: in reasoning mode content_emitted stays 0
            // and the held-back span lives after reason_emitted (tool call before any
            // </think>) — using content_emitted alone would re-emit the whole request's text.
            } else if let Some(held) = crate::tools::held_back_remainder(&acc, reason_emitted.max(content_emitted)) {
                // A tool-call marker was held back but nothing parsed: surface the buffered text
                // as content, exactly what the non-streaming mode returns for the same output.
                let c = content_chunk(&completion_id, created, &model_name, held);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
                content_emitted = acc.len();
            }
            {
                let (r_txt, c_txt) = split_think(&acc, think_open, think_close);
                log_generation(&state.tokenizer, &state.stop_ids, &completion_id, &finish,
                               last_tok, prompt_len, n, r_txt.as_deref(), &c_txt,
                               tool_calls.len(), req_tools.as_ref().map(|t| t.len()).unwrap_or(0),
                               effort_for_log.as_deref(), presence_penalty, pp_source);
            }
            let final_chunk = format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"}}]}}",
                completion_id, created, model_name, spec_finish_reason(&finish));
            if let Some(r) = &otel_req { r.end(&final_chunk); }   // event=stream_end (carries finish_reason)
            yield Ok(Event::default().data(final_chunk));
            if include_usage {
                // Spec stream-usage chunk: empty choices, top-level usage. `timings` rides along
                // as the extension field (strict clients ignore it), and `session_id` echoes the
                // OTel session key so the client can label the stream without computing anything.
                let mut usage_chunk = serde_json::json!({
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name, "choices": [],
                    "usage": {"prompt_tokens": prompt_len, "completion_tokens": n,
                              "total_tokens": prompt_len + n},
                    "timings": make_timings(t0, first_tok, prompt_len, n),
                });
                if let Some(s) = &otel_session { usage_chunk["session_id"] = serde_json::json!(s); }
                yield Ok(Event::default().data(usage_chunk.to_string()));
            }
            // The OpenAI SSE terminator. Without it a strict client sits on an open stream
            // waiting for more events after the finish chunk.
            yield Ok(Event::default().data("[DONE]"));
            let dt = t0.elapsed().as_secs_f32();
            eprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={} stop_hit={}", n, if dt>1e-6 {n as f32/dt} else {0.0}, finish, stop_hit);
        };
        Sse::new(stream).into_response()
    } else {
        eprintln!("[req] sync   prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let t0 = std::time::Instant::now();
        let mut tokens = Vec::new();
        let mut finish = "length".to_string();
        let mut first_tok: Option<std::time::Instant> = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                TokEvent::Tok(t) => {
                    tokens.push(t);
                    if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                    // Apply stop strings LIVE, not just post-hoc: on a hit, break AND let rx drop —
                    // the scheduler sees the closed channel and cancels the lane instead of decoding
                    // to EOS/max_new. Only the tail is searched (a stop string spans a few tokens;
                    // one longer than the window is still honoured post-hoc below, just not early).
                    if !req.stop.is_empty() && tokens.len() % 4 == 0 {
                        let tail = &tokens[tokens.len().saturating_sub(96)..];
                        let s = state.tokenizer.decode(tail, true).unwrap_or_default();
                        if req.stop.iter().any(|x| !x.is_empty() && s.contains(x.as_str())) { break; }
                    }
                }
                TokEvent::Finish { reason } => { finish = reason; break; }
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let mut text = state.tokenizer.decode(&tokens, true).unwrap_or_default();
        if !req.stop.is_empty() {
            if let Some(p) = req.stop.iter().filter_map(|s| text.find(s)).min() {
                text.truncate(p); finish = "stop".to_string();
            }
        }
        eprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={}", tokens.len(), if dt>1e-6 {tokens.len() as f32/dt} else {0.0}, finish);
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let (think_open, think_close, _) = state.tokenizer.think_tags();
        let (reasoning, content) = split_think(&text, think_open, think_close);

        // The model emits calls as <tool_call><function=..><parameter=..>..  -- NOT as JSON. Turn them
        // into OpenAI tool_calls, or the harness just sees XML in the content and never invokes
        // anything. finish_reason MUST become "tool_calls": that is the flag every harness branches on.
        // The (content, tool_calls, finish) DECISION is crate::tools::finalize_parsed — the one
        // canonical serializer shared with the streaming mode, so the two can never diverge again
        // (2026-08-27 user report: a malformed call block vanished in streaming and leaked in JSON).
        // With tools offered, the model's LITERAL output is the only artifact that settles a "the tool
        // ran but nothing happened" report. Log it when asked (RUST_INFER_DUMP_TOOLS=1), and ALWAYS log
        // it when tools were offered and we parsed nothing — that combination means either the model
        // declined, or it emitted a call we failed to understand, and those need very different fixes.
        let parsed = crate::tools::parse(&content, req.tools.as_deref());
        if req.tools.is_some() {
            let dump = std::env::var("RUST_INFER_DUMP_TOOLS").is_ok();
            if dump || parsed.tool_calls.is_empty() {
                eprintln!("[req] raw model output ({} chars): {:?}", content.chars().count(),
                          content.chars().take(1200).collect::<String>());
            }
        }
        // Phase-4 A2: a call to a tool whose schema declares parameters that arrives with EMPTY
        // arguments while the model emitted a NON-EMPTY body is an argument drop, not a quiet
        // success. Log the raw body so the class can never regress unseen again.
        for a in crate::tools::empty_arg_alarms(&content, req.tools.as_deref(), &parsed.tool_calls) {
            eprintln!("[tool-args-alarm] {a}");
        }
        let (content, tool_calls, finish) = crate::tools::finalize_parsed(&content, parsed, &finish);
        if !tool_calls.is_empty() {
            // Log the ARGUMENTS, not just the names. When opencode reported a write as successful and
            // no file appeared, the log said `tool_calls 1: ["write"]` — which is exactly enough to
            // know a tool was called and not nearly enough to know what it was told to do. The path the
            // model chose is the whole question.
            for t in &tool_calls {
                eprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
            }
        }
        let tool_calls = if tool_calls.is_empty() { None } else { Some(tool_calls) };

        log_generation(&state.tokenizer, &state.stop_ids, &completion_id, &finish,
                       tokens.last().copied(), prompt_len, tokens.len(), reasoning.as_deref(),
                       content.as_deref().unwrap_or(""),
                       tool_calls.as_ref().map(|v| v.len()).unwrap_or(0),
                       req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                       effort, presence_penalty, pp_source);
        dump_tokens(&completion_id, &tokens);
        let response = ChatCompletionResponse {
            id: completion_id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model.clone().unwrap_or_else(|| state.model_name.clone()),
            choices: vec![ChatChoice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(), content,
                    reasoning_content: reasoning, tool_calls,
                },
                finish_reason: spec_finish_reason(&finish).to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
            },
            timings: make_timings(t0, first_tok, prompt_len, tokens.len()),
            session_id: otel_session,
        };
        Json(response).into_response()
    }
}
/// Diagnostics-only (env `RUST_INFER_DUMP_TOKENS=1`): one line per generation with the EXACT
/// generated token ids. Why it exists: a served-text comparison cannot distinguish "the same
/// tokens" from "different tokens that detokenize alike", and the bitwise-losslessness claim for a
/// speculative lane (AGENTS §2.6/§3, P14's `--spec-source dflash` A/B) is a statement about the
/// TOKEN sequence. Harness plumbing only — no response byte, no scheduler behaviour is touched.
/// (Twin of `RUST_INFER_DUMP_PROMPT`, which does the same for the prompt ids.)
fn dump_tokens(id: &str, tokens: &[u32]) {
    if std::env::var("RUST_INFER_DUMP_TOKENS").is_err() { return; }
    let ids: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    eprintln!("[gen-ids] id={id} n={} ids=[{}]", tokens.len(), ids.join(","));
}

/// Phase-2 A3 observability: ONE line per generation carrying exactly the fields the Phase-1
/// ledger had to reconstruct — or could not recover at all — from the harness transcripts:
/// the final finish_reason, the terminal token id and whether it is a model stop token, and the
/// reasoning/content token split. `reasoning_tokens`/`content_tokens` are ENCODE-based (the two
/// text spans are re-tokenized), so their sum can differ from `completion_tokens` by
/// template/special-token effects; they are diagnostics, not billing. Log lines only: no
/// response byte, no sampling parameter and no scheduler behaviour is touched.
fn log_generation(tok: &QwenTokenizer, stop_ids: &[u32], id: &str, finish: &str,
                  terminal: Option<u32>, prompt_tokens: usize, completion_tokens: usize,
                  reasoning: Option<&str>, content: &str, tool_calls: usize,
                  tools_offered: usize, effort: Option<&str>, presence_penalty: f32,
                  pp_source: &str) {
    let count = |s: Option<&str>| s.map(|x| tok.encode(x, false).map(|v| v.len()).unwrap_or(0))
        .unwrap_or(0);
    let r_tok = count(reasoning);
    let c_tok = count(Some(content));
    let is_stop = terminal.map(|t| stop_ids.contains(&t)).unwrap_or(false);
    eprintln!("[gen] id={} finish={} terminal_tok={} is_stop_tok={} reasoning_tokens={} \
               content_tokens={} completion_tokens={} prompt_tokens={} tool_calls={} \
               tools_offered={} effort={} presence_penalty={} pp_source={}",
        id, finish, terminal.map(|t| t.to_string()).unwrap_or_else(|| "none".into()), is_stop,
        r_tok, c_tok, completion_tokens, prompt_tokens, tool_calls, tools_offered,
        effort.unwrap_or("<template-default>"), presence_penalty, pp_source);
}

/// Phase-9 A.2 — the status route now carries the engine's OWN per-window decode telemetry
/// (`crate::tel`): mode (mtp|dflash2), tp width, df2 block, chosen depth, accept@k, yield
/// (tokens per verify forward), step p50/p90. Lock-free read; it never touches a decode step.
/// Clients (owner harness, accept_gate.py) use it to see TRUE alpha instead of inferring it
/// from wall-clock tokens. `status` stays "ok" so existing liveness probes are unaffected.
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok", "telemetry": crate::tel::snapshot_json()}))
}

// ─── POST /v1/tokenize ────────────────────────────────────────────────────────────────
// vLLM-compatible de-facto tokenization endpoint. /v1/tokenize has NO OpenAI spec — it is a
// community convention (vLLM, SGLang, llama.cpp's /tokenize, LiteLLM). The OpenAI-spec'd token
// count is the Responses API's /v1/responses/input_tokens/count — a DIFFERENT API family, not
// implemented here. Matched shape (PLAN/ADD_V1_TOKENIZE_PROMPT.md; vLLM's TokenizeRequest/
// TokenizeResponse): response fields {tokens, count, max_model_len}, truncation keeps the LAST n
// tokens (vLLM's left-truncation convention, floor 1), empty prompt -> count 0 (vLLM behavior),
// over-length (>= max_seq_len after truncate) -> 400 `context_length_exceeded` — the same
// threshold the chat path enforces.
//
// PURE TOKENIZER: `QwenTokenizer::encode` (or the chat-template render for `messages`) only —
// no scheduler submit, no forward, no KV, no GPU work. Cheap and synchronous.
//
// Deliberate divergence (owner-pinned contract): stock vLLM defaults `add_special_tokens` to
// FALSE on this endpoint; ours defaults TRUE, mirroring the engine's own serving path
// (chat_completions encodes the rendered template with `true`). The flag's meaning is the HF
// fast-tokenizer POST-PROCESSOR, not the chat template — vLLM never applies a chat template to a
// raw /tokenize prompt either. The shipped Qwen3.5/Hy3/GLM post-processors add nothing for raw
// text (verified against `transformers`), so true==false on those families BY REFERENCE and raw
// counts match vLLM's false default anyway; a tokenizer whose post-processor injects specials
// (Llama-3-style BOS) gets them with `true` (proven by tests/tokenize_golden_test.rs fixture).
async fn tokenize(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Response {
    // The engine's standard error body (same shape as the context_length_exceeded 400 above).
    let bad = |msg: String| -> Response {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": msg, "type": "invalid_request_error", "code": "invalid_request_error",
        }}))).into_response()
    };

    // `model`: required; must name the served model (must match GET /v1/models).
    let Some(model) = body.get("model").and_then(|v| v.as_str()) else {
        return bad("'model' is required and must name the served model (see GET /v1/models)".into());
    };
    if model != state.model_name {
        return bad(format!("model '{model}' not found. Available: {}", state.model_name));
    }

    // `add_special_tokens`: optional bool, default true (see the divergence note above).
    let add_special = match body.get("add_special_tokens") {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => return bad("'add_special_tokens' must be a boolean".into()),
    };

    // `truncate_prompt_tokens`: optional int >= 1 (vLLM's field, vLLM's validation floor).
    let truncate: Option<usize> = match body.get("truncate_prompt_tokens") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Number(n)) => match n.as_u64() {
            Some(v @ 1..) => Some(v as usize),
            _ => return bad("'truncate_prompt_tokens' must be an integer >= 1".into()),
        },
        Some(_) => return bad("'truncate_prompt_tokens' must be an integer >= 1".into()),
    };

    // `prompt` | `messages`: exactly one. Raw text is the vLLM shape; the token-id list is OUR
    // extension (golden/corpus round-trips); `messages` is vLLM's chat-template mode — the model's
    // chat template is applied (generation prompt included), so `count` equals the TRUE prompt
    // size of the equivalent chat request, usage.prompt_tokens included.
    let has_prompt = body.get("prompt").map_or(false, |v| !v.is_null());
    let has_messages = body.get("messages").map_or(false, |v| !v.is_null());
    if has_prompt && has_messages {
        return bad("provide exactly one of 'prompt' or 'messages'".into());
    }
    let mut tokens: Vec<u32> = if has_messages {
        // Messages mode: render EXACTLY as chat_completions does — same ChatMessage deserializer
        // (string or content-array, null content, tool_calls), same optional tools passthrough,
        // same reasoning_effort family normalization — then encode. Any divergence here would show
        // up as tokenize(messages) != usage.prompt_tokens for the same conversation.
        let msgs: Vec<ChatMessage> = match serde_json::from_value(body["messages"].clone()) {
            Ok(m) => m,
            Err(e) => return bad(format!("'messages' is not a valid chat array: {e}")),
        };
        let tools = match body.get("tools") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Array(a)) => Some(a.clone()),
            Some(_) => return bad("'tools' must be an array".into()),
        };
        let effort = resolve_reasoning_effort(&state.tokenizer,
                                              body.get("reasoning_effort").and_then(|v| v.as_str()),
                                              state.reasoning_effort.as_deref());
        // W1: the messages mode of /v1/tokenize renders EXACTLY like chat_completions, including
        // `chat_template_kwargs` — otherwise tokenize(messages) diverges from usage.prompt_tokens.
        let kw = match body.get("chat_template_kwargs") {
            None | Some(serde_json::Value::Null) => None,
            Some(v @ serde_json::Value::Object(_)) => Some(v.clone()),
            Some(_) => return bad("'chat_template_kwargs' must be a JSON object".into()),
        };
        if let Err(e) = crate::tokenizer::enable_thinking_kwarg(kw.as_ref()) {
            return bad(e.to_string());
        }
        let rendered = match state.tokenizer.apply_chat_template(
            &msgs, tools.as_deref(), effort.as_deref(), kw.as_ref(), state.thinking) {
            Ok(p) => p,
            Err(e) => return bad(format!("chat template failed: {e}")),
        };
        // The model's ACTUAL resident tokenizer — token-identity with the reference is the
        // whole point of this endpoint. Never a naive split.
        match state.tokenizer.encode(&rendered, add_special) {
            Ok(t) => t,
            Err(e) => return bad(format!("tokenization failed: {e}")),
        }
    } else {
        match body.get("prompt") {
            // An EMPTY prompt is not an error (vLLM returns count 0): availability probes and
            // count-additivity arithmetic want the empty result, not a 400.
            Some(serde_json::Value::String(text)) => {
                if text.is_empty() {
                    Vec::new()
                } else {
                    match state.tokenizer.encode(text, add_special) {
                        Ok(t) => t,
                        Err(e) => return bad(format!("tokenization failed: {e}")),
                    }
                }
            }
            // Token-id round-trip (our extension per the plan; stock vLLM only accepts a string):
            // echo the ids verbatim. add_special_tokens has nothing to encode here — the ids ARE
            // tokens — so it does not apply (same as vLLM's id-prompt paths elsewhere).
            Some(serde_json::Value::Array(items)) => {
                let vocab = state.tokenizer.vocab_size();
                let mut ids: Vec<u32> = Vec::with_capacity(items.len());
                for it in items {
                    match it.as_u64() {
                        Some(id) if (id as usize) < vocab => ids.push(id as u32),
                        _ => return bad(format!(
                            "'prompt' ids must be integers in [0, {}) (got {it})", vocab)),
                    }
                }
                ids
            }
            _ => return bad("'prompt' must be a string or a list of token ids, or use 'messages'"
                .into()),
        }
    };

    if let Some(n) = truncate {
        if tokens.len() > n {
            tokens.drain(..tokens.len() - n); // keep the LAST n — vLLM's left-truncation convention
        }
    }
    // Over-length: the SAME threshold the chat path enforces (prompt >= max_seq_len has zero
    // generation room and is rejected there), so tokenize(messages) can never bless a prompt the
    // chat request would 400. `code: context_length_exceeded` is the machine-readable signal
    // distinguishing "too long" from transient errors; truncate_prompt_tokens is the escape hatch
    // (it runs BEFORE this check — cap to any n that fits).
    if tokens.len() >= state.max_seq_len {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": format!("This model's maximum context length is {} tokens, but the prompt \
                came to {} tokens. Shorten the input, or pass truncate_prompt_tokens to keep the \
                last n tokens.", state.max_seq_len, tokens.len()),
            "type": "invalid_request_error", "code": "context_length_exceeded",
        }}))).into_response();
    }
    let count = tokens.len();
    eprintln!("[tokenize] model={model} n={count} add_special={add_special} truncate={truncate:?}");
    Json(serde_json::json!({
        "tokens": tokens,
        "count": count,
        "max_model_len": state.max_seq_len,
    }))
    .into_response()
}

// ─── POST /v1/detokenize ──────────────────────────────────────────────────────────────
// vLLM-compatible detokenization: the decode half of the tokenize pair. With BOTH endpoints a
// client can build a prompt of EXACTLY N tokens (encode corpus -> slice N ids -> decode) — the
// llama-bench approach to exact-context benchmarks — instead of converging on a count by
// tokenize->trim->re-tokenize iteration.
//
// Matched shape (vLLM's DetokenizeRequest/DetokenizeResponse):
//   POST {"model": "<served id>",          // required, must match /v1/models (our convention)
//         "tokens": [ids...],              // required; ints in [0, vocab); may be empty -> ""
//         "skip_special_tokens": false}    // optional bool, default FALSE — vLLM's default:
//                                          //   special tokens ARE included in the output text
//   -> 200 {"model": <echo>, "prompt": "<decoded text>"}
//
// PURE TOKENIZER: one whole-sequence `QwenTokenizer::decode` — the byte-level BPE decoder handles
// multi-byte characters split across tokens correctly when the FULL id list is decoded at once
// (the StreamByteDecoder exists only for incremental streaming chunks).
async fn detokenize(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Response {
    let bad = |msg: String| -> Response {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": msg, "type": "invalid_request_error", "code": "invalid_request_error",
        }}))).into_response()
    };

    // `model`: required; must name the served model (must match GET /v1/models).
    let Some(model) = body.get("model").and_then(|v| v.as_str()) else {
        return bad("'model' is required and must name the served model (see GET /v1/models)".into());
    };
    if model != state.model_name {
        return bad(format!("model '{model}' not found. Available: {}", state.model_name));
    }

    // `tokens`: required; a (possibly empty) array of ints within the vocab.
    let Some(items) = body.get("tokens").and_then(|v| v.as_array()) else {
        return bad("'tokens' is required and must be a list of token ids".into());
    };
    let vocab = state.tokenizer.vocab_size();
    let mut ids: Vec<u32> = Vec::with_capacity(items.len());
    for it in items {
        match it.as_u64() {
            Some(id) if (id as usize) < vocab => ids.push(id as u32),
            _ => return bad(format!(
                "'tokens' ids must be integers in [0, {}) (got {it})", vocab)),
        }
    }

    // `skip_special_tokens`: optional bool, default false (vLLM's default: specials INCLUDED).
    let skip_special = match body.get("skip_special_tokens") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => return bad("'skip_special_tokens' must be a boolean".into()),
    };

    let prompt = match state.tokenizer.decode(&ids, skip_special) {
        Ok(p) => p,
        Err(e) => return bad(format!("detokenization failed: {e}")),
    };
    eprintln!("[detokenize] model={model} n={} skip_special={skip_special}", ids.len());
    Json(serde_json::json!({
        "model": model,
        "prompt": prompt,
    }))
    .into_response()
}

/// vLLM-style RAW completion endpoint: the prompt continues verbatim — NO chat template, NO
/// thinking markers. This is the surface llama-benchy (the user's benchmark of record) drives;
/// parity requires it. Also carries `min_tokens`/`ignore_eos` (--exact-tg) end to end.
#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    /// vLLM accepts string | token-id array (token arrays are used verbatim, untokenized).
    #[serde(default)]
    prompt: Option<serde_json::Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    min_tokens: Option<usize>,
    #[serde(default)]
    ignore_eos: Option<bool>,
    #[serde(default)]
    seed: Option<u64>,
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let prompt_tokens: Vec<u32> = match req.prompt.as_ref() {
        Some(serde_json::Value::String(t)) => match state.tokenizer.encode(t, true) {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_REQUEST, format!("tokenize failed: {e}")).into_response(),
        },
        Some(serde_json::Value::Array(a)) if !a.is_empty() && a[0].is_number() =>
            a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect(),
        _ => return (StatusCode::BAD_REQUEST,
                     "prompt must be a string or a non-empty token-id array".to_string()).into_response(),
    };
    let prompt_len = prompt_tokens.len();
    if prompt_len + 8 >= state.max_seq_len {
        return (StatusCode::BAD_REQUEST, format!(
            "prompt {} tokens leaves no room within max_seq_len {}", prompt_len, state.max_seq_len)).into_response();
    }
    let req_max = req.max_tokens.unwrap_or(16)
        .min(state.max_seq_len - prompt_len);
    let (tx, mut rx) = mpsc::unbounded_channel::<TokEvent>();
    let request = BatchRequest {
        prompt: prompt_tokens.clone(),
        max_new: req_max,
        temperature: req.temperature.unwrap_or_else(default_temperature),
        top_p: req.top_p.unwrap_or_else(default_top_p),
        top_k: req.top_k.unwrap_or_else(default_top_k),
        rep_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        min_new: req.min_tokens.unwrap_or(0).min(req_max),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        tx,
        seed: req.seed,
        ckpt_at: None,
        domain: crate::batch::Domain::General,
        received_at: std::time::Instant::now(),
        image_embeds: None,
        image_spans: Vec::new(),
        schema: None,
    };
    let (_mn, _ie) = (request.min_new, request.ignore_eos);
    let _ = state.scheduler.send(request);
    eprintln!("[req] completions prompt_tokens={} max_tokens={} min_tokens={} ignore_eos={} stream={}",
              prompt_len, req_max, _mn, _ie, req.stream);
    let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());
    let cid = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    if req.stream {
        let t0 = std::time::Instant::now();
        let stream = async_stream::stream! {
            let mut ntok: usize = 0;
            let mut first_tok: Option<std::time::Instant> = None;
            while let Some(ev) = rx.recv().await {
                match ev {
                    TokEvent::Tok(t) => {
                        if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                        ntok += 1;
                        let text = state.tokenizer.decode(&[t], true).unwrap_or_default();
                        let chunk = serde_json::json!({
                            "id": cid, "object": "text_completion.chunk", "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "text": text, "finish_reason": null}],
                        });
                        yield Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string()));
                    }
                    TokEvent::Finish { reason } => {
                        let fr = if reason == "length" { "length" } else { "stop" };
                        let chunk = serde_json::json!({
                            "id": cid, "object": "text_completion.chunk", "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "text": "", "finish_reason": fr}],
                        });
                        yield Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string()));
                        yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
                        let dt = t0.elapsed().as_secs_f32();
                        eprintln!("[req] done   completions tok={} ({:.1} tok/s wall) finish={}", ntok, if dt>1e-6 {ntok as f32/dt} else {0.0}, fr);
                        break;
                    }
                }
            }
        };
        return Sse::new(stream).into_response();
    }
    // non-streaming: collect everything, detokenize once, single response.
    let mut toks: Vec<u32> = Vec::with_capacity(req_max);
    let mut finish = "stop".to_string();
    while let Some(ev) = rx.recv().await {
        match ev {
            TokEvent::Tok(t) => toks.push(t),
            TokEvent::Finish { reason } => { finish = if reason == "length" { "length".into() } else { "stop".into() }; break; }
        }
    }
    let text = state.tokenizer.decode(&toks, true).unwrap_or_default();
    dump_tokens(&cid, &toks);
    let json = serde_json::json!({
        "id": cid, "object": "text_completion", "created": created, "model": model_name,
        "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": null}],
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": toks.len(),
                  "total_tokens": prompt_len + toks.len()},
    });
    (StatusCode::OK, axum::Json(json)).into_response()
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:id", get(get_model))
        .route("/health", get(health))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        // Base64 image bodies inflate ~4/3x; a high-res PNG at ~2-4 MB exceeds axum's 2 MB default
        // (""Failed to buffer the request body: length limit exceeded"" on image requests). Raise it
        // so images that other engines accept also arrive here.
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}
