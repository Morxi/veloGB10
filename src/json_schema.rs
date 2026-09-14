//! Host-side JSON-schema constrained decoding (V1 subset) for the OpenAI-compatible API.
//!
//! **Why this module exists.** `response_format: {"type":"json_schema", ...}` used to be *accepted
//! and then silently ignored* — the F6 failure class. A client that sends a schema and gets
//! free-form text back has no way to notice until it parses the answer; the failure looks like a
//! model quality problem, not a server contract violation. This module replaces "silently ignored"
//! with two loud behaviours:
//!
//! 1. [`compile_response_format`] validates the schema and returns `Err(msg)` for anything outside
//!    the V1 subset, **naming the offending keyword**. An unsupported schema can never degrade into
//!    "no constraint applied".
//! 2. [`SchemaMask`] is a byte-level JSON machine plus a token-level allowed-set so the sampler can
//!    mask every token that would leave the schema.
//!
//! **V1 subset:** `type` / `properties` / `required` / `additionalProperties` / `items` /
//! `minItems` / `maxItems` / `minLength` / `maxLength` / `minimum` / `maximum` /
//! `exclusiveMinimum` / `exclusiveMaximum` / `enum` (strings and integers), nesting depth <= 12.
//! Everything else — `pattern`, `format`, `oneOf`/`anyOf`/`allOf`/`not`, `$ref`, `const`,
//! `uniqueItems`, `$defs`, conditionals, unknown keywords — is a hard error.
//!
//! **`additionalProperties` default (deliberate deviation from JSON Schema).** When `properties` is
//! present, an absent `additionalProperties` is treated as **false**; a bare `{"type":"object"}` /
//! `{"type":"json_object"}` (no `properties`) allows any key. JSON Schema's own default is `true`,
//! but an accepted key the caller never declared is indistinguishable from a hallucination, so we
//! take the strict reading and document it here. `"additionalProperties": true` opts back in.
//!
//! **Byte machine semantics.** A document is: optional leading whitespace, one JSON value matching
//! the schema, trailing whitespace, end. The machine is an explicit continuation stack of immutable
//! frames; a *state* is an interned stack snapshot, so states are cheap to compare/deduplicate and
//! `allowed()` results are cacheable per state. `step()` advances by one emitted token; `allowed()`
//! walks the vocabulary trie from the state and prunes branches the moment the machine refuses a
//! byte (a full-vocab rescan per state is never needed).
//!
//! No regex, no new dependencies (std + serde_json), no panics on user input — every rejection is an
//! `Err`/`None`/zeroed bitset.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// Number of `u32` words needed for a `vocab_size`-token bitset (bit `t` = token `t` allowed).
pub fn vocab_words(n: usize) -> usize {
    n / 32 + if n % 32 == 0 { 0 } else { 1 }
}

/// Nesting cap. Beyond this the schema is refused rather than truncated (silent truncation would be
/// a constraint that is not the one the caller asked for).
const MAX_DEPTH: usize = 12;

const SUPPORTED_LIST: &str = "type/properties/required/enum/items/minItems/maxItems/minLength/maxLength/minimum/maximum/exclusiveMinimum/exclusiveMaximum/additionalProperties";

/// Every keyword V1 knows about. Anything else (including the explicitly-unsupported list in the
/// module docs) is refused by name.
const KNOWN_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "enum",
];

const PREFIX: &str = "response_format json_schema: ";

fn err_msg(msg: impl AsRef<str>) -> String {
    format!("{PREFIX}{}", msg.as_ref())
}

fn err_unsupported(kw: &str) -> String {
    err_msg(format!(
        "keyword '{kw}' is not supported in this release (V1 supports: {SUPPORTED_LIST})"
    ))
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ===========================================================================================
// Schema IR + validation
// ===========================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    /// No `type` and no constraint: any JSON value. Also the "additional property" child schema.
    Any,
    Object,
    Array,
    String,
    Integer,
    Number,
    Boolean,
    Null,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::Any => "any",
            Ty::Object => "object",
            Ty::Array => "array",
            Ty::String => "string",
            Ty::Integer => "integer",
            Ty::Number => "number",
            Ty::Boolean => "boolean",
            Ty::Null => "null",
        }
    }
}

/// One compiled schema. Children are indices into the shared table (never `Box`), so machine frames
/// can reference schemas by `u32` and stay `Copy`-cheap inside the state snapshots.
#[derive(Clone, Debug)]
struct Schema {
    ty: Ty,
    props: Vec<(String, u32)>,
    /// Indices into `props`.
    required: Vec<u32>,
    /// `false` = keys outside `props` are rejected. See the module docs for the default.
    additional: bool,
    items: Option<u32>,
    min_items: Option<u64>,
    max_items: Option<u64>,
    min_len: Option<u64>,
    max_len: Option<u64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
    excl_min: Option<f64>,
    excl_max: Option<f64>,
    enum_str: Vec<String>,
    enum_int: Vec<i64>,
    has_enum: bool,
    depth: usize,
}

impl Schema {
    fn any() -> Schema {
        Schema {
            ty: Ty::Any,
            props: Vec::new(),
            required: Vec::new(),
            additional: true,
            items: None,
            min_items: None,
            max_items: None,
            min_len: None,
            max_len: None,
            minimum: None,
            maximum: None,
            excl_min: None,
            excl_max: None,
            enum_str: Vec::new(),
            enum_int: Vec::new(),
            has_enum: false,
            depth: 0,
        }
    }
}

/// Is `kw` meaningful for `ty`? Anything else is refused by name (never ignored).
fn keyword_allowed_for(ty: Ty, kw: &str) -> bool {
    match ty {
        Ty::Object => matches!(kw, "type" | "properties" | "required" | "additionalProperties" | "enum"),
        Ty::Array => matches!(kw, "type" | "items" | "minItems" | "maxItems"),
        Ty::String => matches!(kw, "type" | "minLength" | "maxLength" | "enum"),
        Ty::Integer | Ty::Number => {
            matches!(kw, "type" | "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" | "enum")
        }
        Ty::Boolean | Ty::Null => matches!(kw, "type"),
        // Untyped schemas may only constrain through `enum`; everything else needs an explicit type
        // so we never *infer* a constraint the caller did not state.
        Ty::Any => matches!(kw, "type" | "enum"),
    }
}

fn non_neg_u64(kw: &str, v: &Value) -> Result<u64, String> {
    v.as_u64()
        .ok_or_else(|| err_msg(format!("keyword '{kw}' must be a non-negative integer (found {})", v)))
}

fn num_f64(kw: &str, v: &Value) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| err_msg(format!("keyword '{kw}' must be a number (found {})", v)))
}

/// Compile one schema into `table`, returning its index. `via` names the keyword that pulled this
/// schema in (`"properties"` / `"items"`), used for depth errors.
fn compile_schema(v: &Value, depth: usize, via: &str, table: &mut Vec<Schema>) -> Result<u32, String> {
    if depth > MAX_DEPTH {
        return Err(err_msg(format!(
            "schema is nested deeper than the V1 limit of {MAX_DEPTH} levels (via keyword '{via}')"
        )));
    }
    let obj = v.as_object().ok_or_else(|| {
        err_msg(format!(
            "schema for keyword '{via}' must be a JSON object (found {})",
            type_name(v)
        ))
    })?;

    // Keyword gate FIRST: an unsupported/unknown keyword is refused before its value is interpreted,
    // so the error always names the thing the caller has to change.
    let mut declared: Option<String> = None;
    for k in obj.keys() {
        if !KNOWN_KEYWORDS.contains(&k.as_str()) {
            return Err(err_unsupported(k));
        }
        if k == "type" {
            declared = obj["type"].as_str().map(|s| s.to_string());
        }
    }

    let ty = match obj.get("type") {
        None => Ty::Any,
        Some(Value::String(s)) => match s.as_str() {
            "object" | "json_object" => Ty::Object,
            "array" => Ty::Array,
            "string" => Ty::String,
            "integer" => Ty::Integer,
            "number" => Ty::Number,
            "boolean" => Ty::Boolean,
            "null" => Ty::Null,
            other => {
                return Err(err_msg(format!(
                    "keyword 'type' has unsupported value '{other}' (V1 supports: object/array/string/integer/number/boolean/null/json_object)"
                )))
            }
        },
        Some(Value::Array(_)) => {
            return Err(err_msg(
                "keyword 'type' with a union type array is not supported in this release".to_string(),
            ))
        }
        Some(other) => {
            return Err(err_msg(format!("keyword 'type' must be a string (found {})", type_name(other))))
        }
    };

    let ty_label = match &declared {
        Some(s) => format!("\"type\":\"{s}\""),
        None => "no \"type\"".to_string(),
    };
    for k in obj.keys() {
        if !keyword_allowed_for(ty, k) {
            return Err(err_msg(format!(
                "keyword '{k}' is not supported with {ty_label} in this release (V1 supports: {SUPPORTED_LIST})"
            )));
        }
    }

    let mut s = Schema::any();
    s.ty = ty;
    s.depth = depth;
    // Strict default when a key set is declared; permissive for a bare object/json_object.
    s.additional = !matches!(ty, Ty::Object) || !obj.contains_key("properties");

    match obj.get("additionalProperties") {
        Some(Value::Bool(b)) => s.additional = *b,
        Some(other) => {
            return Err(err_msg(format!(
                "keyword 'additionalProperties' only supports a boolean in this release (found {}); \
                 a subschema form is not supported",
                type_name(other)
            )))
        }
        None => {}
    }

    if let Some(p) = obj.get("properties") {
        let pm = p
            .as_object()
            .ok_or_else(|| err_msg("keyword 'properties' must be an object".to_string()))?;
        for (name, sub) in pm {
            let csi = compile_schema(sub, depth + 1, "properties", table)?;
            s.props.push((name.clone(), csi));
        }
    }

    if let Some(r) = obj.get("required") {
        let arr = r
            .as_array()
            .ok_or_else(|| err_msg("keyword 'required' must be an array of property names".to_string()))?;
        for item in arr {
            let name = item.as_str().ok_or_else(|| {
                err_msg("keyword 'required' must contain only strings".to_string())
            })?;
            let idx = s
                .props
                .iter()
                .position(|(n, _)| n == name)
                .ok_or_else(|| {
                    err_msg(format!(
                        "keyword 'required' names '{name}' which is not declared in 'properties'"
                    ))
                })? as u32;
            if !s.required.contains(&idx) {
                s.required.push(idx);
            }
        }
    }

    if let Some(i) = obj.get("items") {
        let csi = compile_schema(i, depth + 1, "items", table)?;
        s.items = Some(csi);
    }
    if let Some(v) = obj.get("minItems") {
        s.min_items = Some(non_neg_u64("minItems", v)?);
    }
    if let Some(v) = obj.get("maxItems") {
        s.max_items = Some(non_neg_u64("maxItems", v)?);
    }
    if let Some((lo, hi)) = s.min_items.zip(s.max_items) {
        if lo > hi {
            return Err(err_msg(format!("keyword 'minItems' ({lo}) exceeds 'maxItems' ({hi})")));
        }
    }
    if let Some(v) = obj.get("minLength") {
        s.min_len = Some(non_neg_u64("minLength", v)?);
    }
    if let Some(v) = obj.get("maxLength") {
        s.max_len = Some(non_neg_u64("maxLength", v)?);
    }
    if let Some((lo, hi)) = s.min_len.zip(s.max_len) {
        if lo > hi {
            return Err(err_msg(format!("keyword 'minLength' ({lo}) exceeds 'maxLength' ({hi})")));
        }
    }
    if let Some(v) = obj.get("minimum") {
        s.minimum = Some(num_f64("minimum", v)?);
    }
    if let Some(v) = obj.get("maximum") {
        s.maximum = Some(num_f64("maximum", v)?);
    }
    if let Some(v) = obj.get("exclusiveMinimum") {
        s.excl_min = Some(num_f64("exclusiveMinimum", v)?);
    }
    if let Some(v) = obj.get("exclusiveMaximum") {
        s.excl_max = Some(num_f64("exclusiveMaximum", v)?);
    }

    if let Some(e) = obj.get("enum") {
        let arr = e.as_array().ok_or_else(|| {
            err_msg("keyword 'enum' must be an array of allowed values".to_string())
        })?;
        for item in arr {
            match item {
                Value::String(x) => s.enum_str.push(x.clone()),
                Value::Number(n) => match n.as_i64() {
                    Some(i) => s.enum_int.push(i),
                    None => {
                        return Err(err_msg(
                            "keyword 'enum': only string and integer values are supported in this release"
                                .to_string(),
                        ))
                    }
                },
                other => {
                    return Err(err_msg(format!(
                        "keyword 'enum': only string and integer values are supported in this release (found {})",
                        type_name(other)
                    )))
                }
            }
        }
        if s.enum_str.is_empty() && s.enum_int.is_empty() {
            return Err(err_msg("keyword 'enum' must not be empty".to_string()));
        }
        let compatible = match ty {
            Ty::Any => true,
            Ty::String => s.enum_int.is_empty(),
            Ty::Integer | Ty::Number => s.enum_str.is_empty(),
            _ => false,
        };
        if !compatible {
            return Err(err_msg(format!(
                "keyword 'enum' values do not match {ty_label} (V1 supports string and integer enum values only)"
            )));
        }
        s.has_enum = true;
    }

    let idx = table.len() as u32;
    table.push(s);
    Ok(idx)
}

// ===========================================================================================
// Byte machine: string / number / literal sub-states
// ===========================================================================================

/// Escape-sequence position inside a JSON string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Esc {
    None,
    /// Just consumed `\`.
    Bslash,
    /// Inside `\uXXXX`, `n` hex digits collected in `acc`.
    Hex { n: u8, acc: u32 },
    /// High surrogate seen, waiting for `\`.
    WantLowBslash { hi: u32 },
    /// High surrogate seen, waiting for `u`.
    WantLowU { hi: u32 },
    /// Inside the low half `\uYYYY` of a surrogate pair.
    LowHex { n: u8, acc: u32, hi: u32 },
}

/// Incremental JSON-string state. `dec` holds the *decoded* bytes and is only kept when a prefix
/// constraint (enum values, property names) needs it — that keeps the buffers bounded by the
/// longest candidate instead of by the emitted document.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct StrState {
    esc: Esc,
    dec: Vec<u8>,
    cp: u32,
    /// Trailing bytes of a not-yet-complete UTF-8 sequence (a token can split a code point).
    pend: Vec<u8>,
}

impl StrState {
    fn new() -> StrState {
        StrState { esc: Esc::None, dec: Vec::new(), cp: 0, pend: Vec::new() }
    }
    fn is_closed_interior(&self) -> bool {
        self.esc == Esc::None && self.pend.is_empty()
    }
}

enum Step {
    Open,
    Closed,
    Bad,
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn push_cp(st: &mut StrState, cp: u32, keep: bool) -> bool {
    let Some(c) = char::from_u32(cp) else { return false };
    let mut buf = [0u8; 4];
    let enc = c.encode_utf8(&mut buf);
    st.cp += 1;
    if keep {
        st.dec.extend_from_slice(enc.as_bytes());
    }
    true
}

/// Append one raw (non-escape) content byte, validating UTF-8 across token boundaries.
fn push_raw(st: &mut StrState, b: u8, keep: bool) -> bool {
    if b < 0x80 {
        st.cp += 1;
        if keep {
            st.dec.push(b);
        }
        return true;
    }
    st.pend.push(b);
    // The borrow of `pend` ends inside `map`, so the arms may mutate `st`.
    let decoded = std::str::from_utf8(&st.pend).map(|s| (s.chars().count() as u32, s.as_bytes().to_vec()));
    match decoded {
        Ok((n, bytes)) => {
            st.pend.clear();
            st.cp += n;
            if keep {
                st.dec.extend_from_slice(&bytes);
            }
            true
        }
        Err(e) => e.error_len().is_none() && st.pend.len() < 4,
    }
}

/// Advance one byte inside a string. `keep` mirrors the caller's prefix-constraint need.
fn str_advance(st: &mut StrState, b: u8, keep: bool) -> Step {
    match st.esc {
        Esc::None => match b {
            b'"' => Step::Closed,
            b'\\' => {
                st.esc = Esc::Bslash;
                Step::Open
            }
            // Raw control characters must be escaped in JSON.
            0x00..=0x1F => Step::Bad,
            _ => {
                if push_raw(st, b, keep) {
                    Step::Open
                } else {
                    Step::Bad
                }
            }
        },
        Esc::Bslash => {
            let cp = match b {
                b'"' => 0x22,
                b'\\' => 0x5C,
                b'/' => 0x2F,
                b'b' => 0x08,
                b'f' => 0x0C,
                b'n' => 0x0A,
                b'r' => 0x0D,
                b't' => 0x09,
                b'u' => {
                    st.esc = Esc::Hex { n: 0, acc: 0 };
                    return Step::Open;
                }
                _ => return Step::Bad,
            };
            st.esc = Esc::None;
            if push_cp(st, cp, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
        Esc::Hex { n, acc } => {
            let Some(d) = hex_val(b) else { return Step::Bad };
            let acc = acc * 16 + d as u32;
            if n + 1 < 4 {
                st.esc = Esc::Hex { n: n + 1, acc };
                return Step::Open;
            }
            st.esc = Esc::None;
            if (0xD800..=0xDBFF).contains(&acc) {
                st.esc = Esc::WantLowBslash { hi: acc };
                return Step::Open;
            }
            if (0xDC00..=0xDFFF).contains(&acc) {
                // Lone low surrogate: serde_json rejects it too, so we must not emit it.
                return Step::Bad;
            }
            if push_cp(st, acc, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
        Esc::WantLowBslash { hi } => {
            if b != b'\\' {
                return Step::Bad;
            }
            st.esc = Esc::WantLowU { hi };
            Step::Open
        }
        Esc::WantLowU { hi } => {
            if b != b'u' {
                return Step::Bad;
            }
            st.esc = Esc::LowHex { n: 0, acc: 0, hi };
            Step::Open
        }
        Esc::LowHex { n, acc, hi } => {
            let Some(d) = hex_val(b) else { return Step::Bad };
            let acc = acc * 16 + d as u32;
            if n + 1 < 4 {
                st.esc = Esc::LowHex { n: n + 1, acc, hi };
                return Step::Open;
            }
            st.esc = Esc::None;
            if !(0xDC00..=0xDFFF).contains(&acc) {
                return Step::Bad;
            }
            let cp = 0x10000 + ((hi - 0xD800) << 10) + (acc - 0xDC00);
            if push_cp(st, cp, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
    }
}

/// JSON number grammar position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum NumState {
    /// Saw `-`, a digit must follow.
    NegStart,
    Zero,
    Int,
    FracDot,
    Frac,
    ExpChar,
    ExpSign,
    Exp,
}

impl NumState {
    fn terminal(self) -> bool {
        matches!(self, NumState::Zero | NumState::Int | NumState::Frac | NumState::Exp)
    }
}

fn num_start(b: u8) -> Option<(NumState, Vec<u8>)> {
    match b {
        b'-' => Some((NumState::NegStart, vec![b'-'])),
        b'0' => Some((NumState::Zero, vec![b'0'])),
        b'1'..=b'9' => Some((NumState::Int, vec![b])),
        _ => None,
    }
}

fn num_advance(st: NumState, b: u8) -> Option<NumState> {
    use NumState::*;
    match (st, b) {
        (NegStart, b'0') => Some(Zero),
        (NegStart, b'1'..=b'9') => Some(Int),
        (Zero, b'.') => Some(FracDot),
        (Zero, b'e' | b'E') => Some(ExpChar),
        (Int, b'0'..=b'9') => Some(Int),
        (Int, b'.') => Some(FracDot),
        (Int, b'e' | b'E') => Some(ExpChar),
        (FracDot, b'0'..=b'9') => Some(Frac),
        (Frac, b'0'..=b'9') => Some(Frac),
        (Frac, b'e' | b'E') => Some(ExpChar),
        (ExpChar, b'0'..=b'9') => Some(Exp),
        (ExpChar, b'+' | b'-') => Some(ExpSign),
        (ExpSign, b'0'..=b'9') => Some(Exp),
        (Exp, b'0'..=b'9') => Some(Exp),
        _ => None,
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

// ===========================================================================================
// Byte machine: frames + states
// ===========================================================================================

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ObjPhase {
    /// Whitespace, then `}` or the opening quote of a key.
    KeyOrEnd,
    /// Directly after a `,`: a key must follow (`}` would be a trailing comma).
    KeyReq,
    /// Inside a key string. Parsed *inside* the object frame so phase transitions never need to
    /// reach across stack frames.
    Key(StrState),
    /// Key closed; expect whitespace then `:`. `child` is the schema for the member value.
    Colon { child: u32 },
    /// The member value has been parsed; expect whitespace, `,` or `}`.
    AfterValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum ArrPhase {
    /// Whitespace, then `]` or the start of an item.
    ItemOrEnd,
    /// Directly after a `,`: an item must follow (`]` would be a trailing comma).
    ItemReq,
    /// An item has been parsed; expect whitespace, `,` or `]`.
    AfterItem,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Frame {
    /// Top-level frame: after the value only whitespace (then end) may follow.
    DocEnd,
    /// The byte stream left the schema / the JSON grammar. Absorbs everything after it.
    Dead,
    /// A value matching schema `si` must start here.
    Value { si: u32 },
    Obj { si: u32, phase: ObjPhase, seen: Vec<u32>, extra: Vec<String> },
    Arr { si: u32, n: u64, phase: ArrPhase },
    Str { si: u32, st: StrState },
    /// Remaining bytes of `true` / `false` / `null`.
    Lit { rest: &'static [u8] },
    Num { si: u32, buf: Vec<u8>, st: NumState },
}

/// An interned machine state: a stack of continuation frames.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
struct Machine {
    stack: Vec<Frame>,
}

impl Machine {
    fn is_dead(&self) -> bool {
        self.stack.iter().any(|f| matches!(f, Frame::Dead))
    }
    /// Exactly the completed-document shape: only `DocEnd` left (whitespace is absorbed in place).
    fn at_doc_end(&self) -> bool {
        self.stack.len() == 1 && matches!(self.stack[0], Frame::DocEnd)
    }
}

/// Undo log for a single byte, used by the trie walk to roll a shared machine back instead of
/// cloning it per node. Frames are only ever mutated *after* being popped, so logging the frames
/// that were popped (first occurrence per index) plus the deepest index touched is a complete
/// record of the change.
struct Roll {
    orig_len: usize,
    frames: Vec<(usize, Frame)>,
}

impl Roll {
    fn new(orig_len: usize) -> Roll {
        Roll { orig_len, frames: Vec::new() }
    }
    fn note_pop(&mut self, idx: usize, f: &Frame) {
        if idx >= self.orig_len {
            // Pushed earlier during this same byte — truncation removes it.
            return;
        }
        if self.frames.iter().any(|(i, _)| *i == idx) {
            return;
        }
        self.frames.push((idx, f.clone()));
    }
    fn restore(&self, m: &mut Machine) {
        if self.frames.is_empty() {
            return;
        }
        let base = self.frames.iter().map(|(i, _)| *i).min().unwrap_or(0);
        m.stack.truncate(base);
        let mut sorted: Vec<&(usize, Frame)> = self.frames.iter().collect();
        sorted.sort_by_key(|(i, _)| *i);
        for (_, f) in sorted {
            m.stack.push(f.clone());
        }
    }
}

// ===========================================================================================
// Vocabulary trie
// ===========================================================================================

#[derive(Default, Debug)]
struct TrieNode {
    children: Vec<(u8, usize)>,
    token: Option<u32>,
}

#[derive(Default, Debug)]
struct Vocab {
    nodes: Vec<TrieNode>,
    by_id: HashMap<u32, Vec<u8>>,
}

impl Vocab {
    /// Build the byte trie. Empty pieces are skipped: a token with no bytes can never advance the
    /// machine, so it is never allowed (see `step`).
    fn build(pieces: Vec<(u32, Vec<u8>)>) -> (Vocab, usize) {
        let mut v = Vocab::default();
        v.nodes.push(TrieNode::default());
        let mut vocab_size = 0usize;
        for (id, bytes) in pieces {
            vocab_size = vocab_size.max(id as usize + 1);
            if bytes.is_empty() {
                continue;
            }
            v.by_id.insert(id, bytes.clone());
            let mut cur = 0usize;
            for &b in &bytes {
                let nxt = match v.nodes[cur].children.iter().find(|(c, _)| *c == b) {
                    Some(&(_, n)) => n,
                    None => {
                        v.nodes.push(TrieNode::default());
                        let n = v.nodes.len() - 1;
                        v.nodes[cur].children.push((b, n));
                        n
                    }
                };
                cur = nxt;
            }
            v.nodes[cur].token = Some(id); // duplicate piece: last id wins
        }
        (v, vocab_size)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A poisoned lock means a previous panic; the bitsets/states are still structurally valid, so
    // keep serving instead of panicking a second time.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Default, Debug)]
struct Interner {
    states: Vec<Arc<Machine>>,
    index: HashMap<Machine, u32>,
}

// ===========================================================================================
// Public API
// ===========================================================================================

/// A compiled schema: byte-level machine + token-level allowed sets, lazily built.
#[derive(Debug)]
pub struct SchemaMask {
    schemas: Vec<Schema>,
    root: u32,
    /// Index 0 is always the universal permissive schema (`Ty::Any`), used for additional
    /// properties and for `{"type":"array"}` without `items`.
    start: u32,
    interner: Mutex<Interner>,
    /// `state -> allowed bitset`. Computed on demand; states are interned so identical machine
    /// snapshots share one entry.
    cache: Mutex<HashMap<u32, Arc<Vec<u32>>>>,
    vocab: Vocab,
    vocab_size: usize,
    words: usize,
}

impl SchemaMask {
    /// The start state id.
    pub fn start_state(&self) -> u32 {
        self.start
    }

    /// Allowed-token bitset for `state`: `vocab_words(vocab_size)` little-endian `u32` words,
    /// bit `t` set = token `t` is allowed. Cached; safe to call from several threads.
    ///
    /// Returns an all-zero bitset if `set_vocab` was never called (`fail closed`: nothing is
    /// emitted rather than unconstrained text).
    pub fn allowed(&self, state: u32) -> Arc<Vec<u32>> {
        if let Some(hit) = lock(&self.cache).get(&state).cloned() {
            return hit;
        }
        let bits = self.compute_allowed(state);
        let arc = Arc::new(bits);
        lock(&self.cache).insert(state, arc.clone());
        arc
    }

    /// Advance the machine by ONE EMITTED TOKEN. `None` when the token is unknown, empty, or its
    /// bytes are refused in `state`.
    pub fn step(&self, state: u32, tok: u32) -> Option<u32> {
        let bytes = self.vocab.by_id.get(&tok)?;
        if bytes.is_empty() {
            return None;
        }
        let mut m = (*self.machine(state)?).clone();
        if m.is_dead() {
            return None;
        }
        for &b in bytes {
            if !self.feed(&mut m, b, None) {
                return None;
            }
        }
        Some(self.intern(m))
    }

    /// True when the JSON value is finished and only trailing whitespace (or end) may follow.
    /// A pending number is finalized here, so `42` counts as finished even though the frame is
    /// still open until the next byte.
    pub fn is_complete(&self, state: u32) -> bool {
        let Some(stored) = self.machine(state) else { return false };
        if stored.is_dead() {
            return false;
        }
        let mut m = (*stored).clone();
        if !self.finalize_pending_number(&mut m) {
            return false;
        }
        m.at_doc_end()
    }

    /// True when a state is DEAD (used to reject after a broken draft chain).
    pub fn is_dead(&self, state: u32) -> bool {
        self.machine(state).map(|m| m.is_dead()).unwrap_or(true)
    }

    /// Set the token-id → piece-bytes table (built from the model tokenizer). MUST be called once
    /// before `allowed()`; also fixes `vocab_size`/`words`. Cached bitsets are dropped because they
    /// were computed against the previous vocabulary.
    pub fn set_vocab(&mut self, pieces: Vec<(u32, Vec<u8>)>) {
        let (v, size) = Vocab::build(pieces);
        self.vocab = v;
        self.vocab_size = size;
        self.words = vocab_words(size);
        lock(&self.cache).clear();
    }

    /// A bitset with every token disallowed EXCEPT the model's end-of-generation tokens: when the
    /// document is complete, generation must stop rather than run on. End tokens are deliberately
    /// never produced by `allowed()`.
    pub fn end_mask(&self, stop_ids: &[u32]) -> Arc<Vec<u32>> {
        let words = if self.words > 0 {
            self.words
        } else {
            // Vocab not attached yet: size the bitset to the stop ids so the caller still gets a
            // usable mask instead of an empty one.
            vocab_words(stop_ids.iter().map(|&s| s as usize + 1).max().unwrap_or(0))
        };
        let mut bits = vec![0u32; words];
        for &s in stop_ids {
            let t = s as usize;
            if t < words * 32 {
                bits[t >> 5] |= 1u32 << (t & 31);
            }
        }
        Arc::new(bits)
    }

    /// Human-readable summary for logs.
    pub fn summary(&self) -> String {
        let root = &self.schemas[self.root as usize];
        let max_depth = self.schemas.iter().map(|s| s.depth).max().unwrap_or(0);
        let enum_values: usize = self.schemas.iter().map(|s| s.enum_str.len() + s.enum_int.len()).sum();
        let states = lock(&self.interner).states.len();
        let cached = lock(&self.cache).len();
        let vocab = if self.vocab_size == 0 {
            "unset".to_string()
        } else {
            format!("{} tokens/{} words", self.vocab_size, self.words)
        };
        format!(
            "json_schema: root={} properties={} required={} additionalProperties={} items={} enum_values={} schemas={} max_depth={} states={} cached_masks={} vocab={}",
            root.ty.name(),
            root.props.len(),
            root.required.len(),
            root.additional,
            root.items.is_some(),
            enum_values,
            self.schemas.len(),
            max_depth,
            states,
            cached,
            vocab,
        )
    }

    // -----------------------------------------------------------------------------------
    // internals
    // -----------------------------------------------------------------------------------

    fn machine(&self, state: u32) -> Option<Arc<Machine>> {
        lock(&self.interner).states.get(state as usize).cloned()
    }

    fn intern(&self, m: Machine) -> u32 {
        let mut it = lock(&self.interner);
        if let Some(&id) = it.index.get(&m) {
            return id;
        }
        let id = it.states.len() as u32;
        it.index.insert(m.clone(), id);
        it.states.push(Arc::new(m));
        id
    }

    fn compute_allowed(&self, state: u32) -> Vec<u32> {
        let mut bits = vec![0u32; self.words];
        if self.words == 0 || self.vocab.nodes.is_empty() {
            return bits;
        }
        let Some(stored) = self.machine(state) else { return bits };
        if stored.is_dead() {
            return bits;
        }
        let mut m = (*stored).clone();
        self.trie_walk(&mut m, 0, &mut bits);
        bits
    }

    /// DFS the vocabulary trie from `node`, pruning the moment the machine refuses a byte.
    /// One machine is reused and rolled back per byte (see [`Roll`]) — no per-node clones and no
    /// full-vocab rescan.
    fn trie_walk(&self, m: &mut Machine, node: usize, out: &mut Vec<u32>) {
        let children = self.vocab.nodes[node].children.clone();
        for (b, child) in children {
            let mut roll = Roll::new(m.stack.len());
            let alive = self.feed(m, b, Some(&mut roll));
            if alive {
                if let Some(tok) = self.vocab.nodes[child].token {
                    let t = tok as usize;
                    if t < self.vocab_size {
                        out[t >> 5] |= 1u32 << (t & 31);
                    }
                }
                self.trie_walk(m, child, out);
            }
            roll.restore(m);
        }
    }

    /// Push the frame(s) that start a JSON value matching `si` in response to byte `b`.
    fn value_start(&self, m: &mut Machine, si: u32, b: u8) -> bool {
        let s = &self.schemas[si as usize];
        let (has_enum, any_str, any_int) = (s.has_enum, !s.enum_str.is_empty(), !s.enum_int.is_empty());
        let push_str = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Str { si, st: StrState::new() });
        };
        let push_obj = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Obj {
                si,
                phase: ObjPhase::KeyOrEnd,
                seen: Vec::new(),
                extra: Vec::new(),
            });
        };
        let push_arr = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Arr { si, n: 0, phase: ArrPhase::ItemOrEnd });
        };
        match s.ty {
            Ty::Object => {
                if b != b'{' {
                    return false;
                }
                push_obj(m, si);
                true
            }
            Ty::Array => {
                if b != b'[' {
                    return false;
                }
                push_arr(m, si);
                true
            }
            Ty::String => {
                if b != b'"' {
                    return false;
                }
                push_str(m, si);
                true
            }
            Ty::Integer | Ty::Number => match num_start(b) {
                Some((st, buf)) => {
                    m.stack.push(Frame::Num { si, buf, st });
                    true
                }
                None => false,
            },
            Ty::Boolean => match b {
                b't' => {
                    m.stack.push(Frame::Lit { rest: b"rue" });
                    true
                }
                b'f' => {
                    m.stack.push(Frame::Lit { rest: b"alse" });
                    true
                }
                _ => false,
            },
            Ty::Null => {
                if b != b'n' {
                    return false;
                }
                m.stack.push(Frame::Lit { rest: b"ull" });
                true
            }
            Ty::Any => {
                // Untyped: accept whatever the schema constrains. With an enum, only the value
                // kinds the enum actually contains may start.
                if has_enum && !any_str && b == b'"' {
                    return false;
                }
                if has_enum && !any_int && (b == b'-' || b.is_ascii_digit()) {
                    return false;
                }
                match b {
                    b'{' => {
                        push_obj(m, si);
                        true
                    }
                    b'[' => {
                        push_arr(m, si);
                        true
                    }
                    b'"' => {
                        push_str(m, si);
                        true
                    }
                    b't' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"rue" });
                        true
                    }
                    b'f' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"alse" });
                        true
                    }
                    b'n' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"ull" });
                        true
                    }
                    _ => match num_start(b) {
                        Some((st, buf)) => {
                            m.stack.push(Frame::Num { si, buf, st });
                            true
                        }
                        None => false,
                    },
                }
            }
        }
    }

    fn required_satisfied(&self, si: u32, seen: &[u32]) -> bool {
        self.schemas[si as usize].required.iter().all(|r| seen.contains(r))
    }

    /// Value-string constraint check. `closing` = the terminating quote was consumed.
    fn str_value_ok(&self, si: u32, st: &StrState, closing: bool) -> bool {
        let s = &self.schemas[si as usize];
        if let Some(mx) = s.max_len {
            if st.cp as u64 > mx {
                return false;
            }
        }
        if closing {
            if !st.is_closed_interior() {
                return false;
            }
            if let Some(mn) = s.min_len {
                if (st.cp as u64) < mn {
                    return false;
                }
            }
            if !s.enum_str.is_empty() && !s.enum_str.iter().any(|e| e.as_bytes() == st.dec.as_slice()) {
                return false;
            }
        } else if !s.enum_str.is_empty()
            && !s.enum_str.iter().any(|e| e.as_bytes().starts_with(&st.dec))
        {
            // enum prefix rule: only bytes that keep us on the way to an allowed value.
            return false;
        }
        true
    }

    fn num_ok(&self, si: u32, buf: &[u8], st: NumState) -> bool {
        if !st.terminal() {
            return false;
        }
        let s = &self.schemas[si as usize];
        let Ok(text) = std::str::from_utf8(buf) else { return false };
        let Ok(v) = text.parse::<f64>() else { return false };
        if s.ty == Ty::Integer && !matches!(st, NumState::Zero | NumState::Int) {
            return false; // integer forbids fraction/exponent
        }
        if s.has_enum {
            if s.enum_str.is_empty() {
                if !s.enum_int.iter().any(|&e| e as f64 == v) {
                    return false;
                }
            } else {
                return false; // string enum on a numeric schema cannot match
            }
        }
        if let Some(m) = s.minimum {
            if v < m {
                return false;
            }
        }
        if let Some(m) = s.excl_min {
            if v <= m {
                return false;
            }
        }
        if let Some(m) = s.maximum {
            if v > m {
                return false;
            }
        }
        if let Some(m) = s.excl_max {
            if v >= m {
                return false;
            }
        }
        true
    }

    /// Pop a terminal pending number frame (used by `is_complete`); false = it was not valid.
    fn finalize_pending_number(&self, m: &mut Machine) -> bool {
        if let Some(Frame::Num { si, buf, st }) = m.stack.last().cloned() {
            if self.num_ok(si, &buf, st) {
                m.stack.pop();
                return true;
            }
            return false;
        }
        true
    }

    fn key_prefix_ok(&self, si: u32, dec: &[u8]) -> bool {
        let s = &self.schemas[si as usize];
        s.additional || s.props.iter().any(|(n, _)| n.as_bytes().starts_with(dec))
    }

    /// Feed ONE byte. Returns true when the byte was consumed by a live machine; false when the
    /// machine died (a [`Frame::Dead`] is pushed and absorbs everything afterwards).
    ///
    /// Frames are mutated only after being popped, which is what makes the [`Roll`] undo log and
    /// the "pop, then let the parent reprocess the same byte" loop sound.
    fn feed(&self, m: &mut Machine, b: u8, mut roll: Option<&mut Roll>) -> bool {
        loop {
            if m.stack.is_empty() {
                m.stack.push(Frame::Dead);
                return false;
            }
            let idx = m.stack.len() - 1;
            let top = m.stack.pop().expect("checked non-empty");
            if let Some(r) = roll.as_deref_mut() {
                r.note_pop(idx, &top);
            }
            return match top {
                Frame::Dead => {
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::DocEnd => {
                    if is_ws(b) {
                        m.stack.push(Frame::DocEnd);
                        return true;
                    }
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::Value { si } => {
                    if is_ws(b) {
                        m.stack.push(Frame::Value { si });
                        return true;
                    }
                    if self.value_start(m, si, b) {
                        return true;
                    }
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::Obj { si, phase, mut seen, mut extra } => match phase {
                    ObjPhase::KeyOrEnd => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyOrEnd, seen, extra });
                            return true;
                        }
                        if b == b'}' {
                            if self.required_satisfied(si, &seen) {
                                return true; // object closed; the byte belongs to this frame
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        if b == b'"' {
                            m.stack.push(Frame::Obj {
                                si,
                                phase: ObjPhase::Key(StrState::new()),
                                seen,
                                extra,
                            });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    ObjPhase::Key(mut st) => {
                        match str_advance(&mut st, b, true) {
                            Step::Open => {
                                if !self.key_prefix_ok(si, &st.dec) {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                }
                                m.stack.push(Frame::Obj { si, phase: ObjPhase::Key(st), seen, extra });
                                true
                            }
                            Step::Closed => {
                                if !st.is_closed_interior() {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                }
                                let Ok(key) = String::from_utf8(st.dec.clone()) else {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                };
                                let s = &self.schemas[si as usize];
                                let child = match s.props.iter().position(|(n, _)| *n == key) {
                                    Some(i) => {
                                        if seen.contains(&(i as u32)) {
                                            m.stack.push(Frame::Dead);
                                            return false;
                                        }
                                        seen.push(i as u32);
                                        s.props[i].1
                                    }
                                    None => {
                                        if !s.additional || extra.iter().any(|k| *k == key) {
                                            m.stack.push(Frame::Dead);
                                            return false;
                                        }
                                        extra.push(key);
                                        0 // universal permissive schema
                                    }
                                };
                                m.stack.push(Frame::Obj {
                                    si,
                                    phase: ObjPhase::Colon { child },
                                    seen,
                                    extra,
                                });
                                true
                            }
                            Step::Bad => {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                    }
                    // After a `,` a key must follow: `}` here would be a trailing comma (invalid
                    // JSON), so it can never be emitted.
                    ObjPhase::KeyReq => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyReq, seen, extra });
                            return true;
                        }
                        if b == b'"' {
                            m.stack.push(Frame::Obj {
                                si,
                                phase: ObjPhase::Key(StrState::new()),
                                seen,
                                extra,
                            });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    ObjPhase::Colon { child } => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::Colon { child }, seen, extra });
                            return true;
                        }
                        if b == b':' {
                            // Consume the separator, then require a value above us.
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::AfterValue, seen, extra });
                            m.stack.push(Frame::Value { si: child });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                    ObjPhase::AfterValue => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::AfterValue, seen, extra });
                            return true;
                        }
                        if b == b',' {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyReq, seen, extra });
                            return true;
                        }
                        if b == b'}' {
                            if self.required_satisfied(si, &seen) {
                                return true;
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                },
                Frame::Arr { si, n, phase } => match phase {
                    ArrPhase::ItemOrEnd | ArrPhase::ItemReq => {
                        // Just after a `,` (`ItemReq`) a `]` would be a trailing comma: refuse it.
                        let need_item = matches!(phase, ArrPhase::ItemReq);
                        if is_ws(b) {
                            let here = if need_item { ArrPhase::ItemReq } else { ArrPhase::ItemOrEnd };
                            m.stack.push(Frame::Arr { si, n, phase: here });
                            return true;
                        }
                        if b == b']' {
                            let s = &self.schemas[si as usize];
                            if !need_item && s.min_items.map_or(true, |m0| n >= m0) {
                                return true; // array closed
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        let s = &self.schemas[si as usize];
                        if let Some(mx) = s.max_items {
                            if n >= mx {
                                m.stack.push(Frame::Dead);
                                return false;
                            }
                        }
                        let item_si = s.items.unwrap_or(0);
                        m.stack.push(Frame::Arr { si, n: n + 1, phase: ArrPhase::AfterItem });
                        m.stack.push(Frame::Value { si: item_si });
                        continue; // same byte starts the item
                    }
                    ArrPhase::AfterItem => {
                        if is_ws(b) {
                            m.stack.push(Frame::Arr { si, n, phase: ArrPhase::AfterItem });
                            return true;
                        }
                        if b == b',' {
                            m.stack.push(Frame::Arr { si, n, phase: ArrPhase::ItemReq });
                            return true;
                        }
                        if b == b']' {
                            let s = &self.schemas[si as usize];
                            if s.min_items.map_or(true, |m0| n >= m0) {
                                return true;
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                },
                Frame::Str { si, mut st } => {
                    let keep = !self.schemas[si as usize].enum_str.is_empty();
                    match str_advance(&mut st, b, keep) {
                        Step::Open => {
                            if self.str_value_ok(si, &st, false) {
                                m.stack.push(Frame::Str { si, st });
                                true
                            } else {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                        Step::Closed => {
                            if self.str_value_ok(si, &st, true) {
                                true
                            } else {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                        Step::Bad => {
                            m.stack.push(Frame::Dead);
                            false
                        }
                    }
                }
                Frame::Lit { rest } => {
                    if rest.is_empty() || rest[0] != b {
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    if rest.len() == 1 {
                        return true; // literal complete
                    }
                    m.stack.push(Frame::Lit { rest: &rest[1..] });
                    true
                }
                Frame::Num { si, mut buf, st } => {
                    if let Some(next) = num_advance(st, b) {
                        buf.push(b);
                        m.stack.push(Frame::Num { si, buf, st: next });
                        return true;
                    }
                    // The number cannot continue: validate it, then hand the byte to the parent.
                    if self.num_ok(si, &buf, st) {
                        continue;
                    }
                    m.stack.push(Frame::Dead);
                    false
                }
            };
        }
    }
}

/// Parse + validate an OpenAI `response_format` value (the whole `{"type": "json_schema", ...}`
/// object).
///
/// Returns `Ok(None)` when there is no schema to enforce (absent/`Null`/`{"type":"text"}`), and
/// `Err(msg)` with a LOUD, user-facing message naming the offending keyword for everything else —
/// an unsupported schema is refused, never silently ignored.
pub fn compile_response_format(rf: &serde_json::Value) -> Result<Option<SchemaMask>, String> {
    let obj = match rf {
        Value::Null => return Ok(None),
        Value::Object(o) => o,
        other => {
            return Err(err_msg(format!(
                "response_format must be an object (found {})",
                type_name(other)
            )))
        }
    };

    let ty = match obj.get("type") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(err_msg(format!(
                "response_format 'type' must be a string (found {})",
                type_name(other)
            )))
        }
        None => return Err(err_msg("response_format is missing 'type'")),
    };

    match ty.as_str() {
        "text" => {
            for k in obj.keys() {
                if k != "type" {
                    return Err(err_msg(format!(
                        "response_format field '{k}' is not supported in this release"
                    )));
                }
            }
            Ok(None)
        }
        "json_object" => {
            for k in obj.keys() {
                if k != "type" {
                    return Err(err_msg(format!(
                        "response_format field '{k}' is not supported in this release"
                    )));
                }
            }
            // `{"type":"json_object"}` means "any JSON object": compile a real permissive object
            // schema, so arrays/scalars are rejected instead of any value being accepted.
            Ok(Some(build_mask(&serde_json::json!({"type": "object"}), 1, "type")?))
        }
        "json_schema" => {
            for k in obj.keys() {
                if k != "type" && k != "json_schema" && k != "schema" {
                    return Err(err_msg(format!(
                        "response_format field '{k}' is not supported in this release"
                    )));
                }
            }
            let inner = obj.get("json_schema");
            let (schema_v, name) = match inner {
                Some(Value::Object(io)) => {
                    for k in io.keys() {
                        if !matches!(k.as_str(), "name" | "description" | "strict" | "schema") {
                            return Err(err_msg(format!(
                                "json_schema field '{k}' is not supported in this release \
                                 (supported: name/description/strict/schema)"
                            )));
                        }
                    }
                    let name = match io.get("name") {
                        Some(Value::String(n)) => Some(n.clone()),
                        Some(other) => {
                            return Err(err_msg(format!(
                                "json_schema 'name' must be a string (found {})",
                                type_name(other)
                            )))
                        }
                        None => None,
                    };
                    (io.get("schema"), name)
                }
                Some(other) => {
                    return Err(err_msg(format!(
                        "'json_schema' must be an object (found {})",
                        type_name(other)
                    )))
                }
                None => (obj.get("schema"), None),
            };
            let Some(sv) = schema_v else {
                return Err(err_msg(
                    "keyword 'schema' is required: send \
                     {\"type\":\"json_schema\",\"json_schema\":{\"name\":...,\"schema\":{...}}}",
                ));
            };
            let what = name.map(|n| format!("json_schema '{n}'")).unwrap_or_else(|| "json_schema".to_string());
            let mask = build_mask(sv, 1, "schema")?;
            // Loud success line: a wrong-but-accepted schema must be visible in the server log.
            eprintln!("[json_schema] enforcing {what}: {}", mask.summary());
            Ok(Some(mask))
        }
        other => Err(err_msg(format!(
            "response_format 'type' = '{other}' is not supported in this release \
             (V1 supports: text/json_object/json_schema)"
        ))),
    }
}

fn build_mask(schema: &Value, depth: usize, via: &str) -> Result<SchemaMask, String> {
    let mut table = vec![Schema::any()]; // index 0 = universal permissive schema
    // Validation errors MUST reach the caller: swallowing them here would turn an unsupported
    // schema into a silently-unenforced one — exactly the failure this module exists to remove.
    let root = compile_schema(schema, depth, via, &mut table)?;
    let mut m = SchemaMask {
        schemas: table,
        root,
        start: 0,
        interner: Mutex::new(Interner::default()),
        cache: Mutex::new(HashMap::new()),
        vocab: Vocab::default(),
        vocab_size: 0,
        words: 0,
    };
    let start = m.intern(Machine { stack: vec![Frame::DocEnd, Frame::Value { si: root }] });
    m.start = start;
    Ok(m)
}

// ===========================================================================================
// Tests
// ===========================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_vocab() -> Vec<(u32, Vec<u8>)> {
        (0u32..256).map(|b| (b, vec![b as u8])).collect()
    }

    fn compile_err(schema_json: &str) -> String {
        let v: Value = serde_json::from_str(schema_json).expect("test schema is valid JSON");
        let rf = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "strict": true, "schema": v}
        });
        compile_response_format(&rf).expect_err("schema must be refused")
    }

    fn mask(schema_json: &str) -> SchemaMask {
        let v: Value = serde_json::from_str(schema_json).expect("test schema is valid JSON");
        let rf = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "strict": true, "schema": v}
        });
        let mut m = compile_response_format(&rf)
            .unwrap_or_else(|e| panic!("schema unexpectedly refused: {e}"))
            .expect("schema must be enforced");
        m.set_vocab(byte_vocab());
        m
    }

    /// Feed a byte string as a sequence of one-byte tokens; `None` = refused.
    fn walk(m: &SchemaMask, s: &str) -> Option<u32> {
        let mut st = m.start_state();
        for &b in s.as_bytes() {
            st = m.step(st, b as u32)?;
        }
        Some(st)
    }

    /// Full-document acceptance: every byte feeds AND the document is finished.
    fn ok(m: &SchemaMask, s: &str) -> bool {
        match walk(m, s) {
            Some(st) => m.is_complete(st),
            None => false,
        }
    }

    fn bits(ids: &[u32], words: usize) -> Vec<u32> {
        let mut v = vec![0u32; words];
        for &i in ids {
            v[(i >> 5) as usize] |= 1u32 << (i & 31);
        }
        v
    }

    // ---- 1: every rejected keyword is named loudly ------------------------------------------

    #[test]
    fn rejected_keywords_are_named_loudly() {
        let cases: &[(&str, &str)] = &[
            ("pattern", r#"{"type":"string","pattern":"^a$"}"#),
            ("format", r#"{"type":"string","format":"date-time"}"#),
            ("oneOf", r#"{"oneOf":[{"type":"string"}]}"#),
            ("anyOf", r#"{"anyOf":[{"type":"string"}]}"#),
            ("allOf", r#"{"allOf":[{"type":"string"}]}"#),
            ("not", r#"{"not":{"type":"string"}}"#),
            ("$ref", r##"{"$ref":"#/defs/x"}"##),
            ("patternProperties", r#"{"type":"object","patternProperties":{"a":{"type":"string"}}}"#),
            ("minProperties", r#"{"type":"object","minProperties":1}"#),
            ("maxProperties", r#"{"type":"object","maxProperties":3}"#),
            ("$defs", r#"{"$defs":{"a":{"type":"string"}}}"#),
            ("definitions", r#"{"definitions":{"a":{"type":"string"}}}"#),
            ("if", r#"{"if":{"type":"string"}}"#),
            ("then", r#"{"then":{"type":"string"}}"#),
            ("else", r#"{"else":{"type":"string"}}"#),
            ("const", r#"{"const":1}"#),
            ("uniqueItems", r#"{"type":"array","uniqueItems":true}"#),
            ("contains", r#"{"type":"array","contains":{"type":"string"}}"#),
            ("prefixItems", r#"{"type":"array","prefixItems":[{"type":"string"}]}"#),
            ("additionalItems", r#"{"type":"array","additionalItems":false}"#),
            ("propertyNames", r#"{"type":"object","propertyNames":{"type":"string"}}"#),
            ("dependencies", r#"{"type":"object","dependencies":{"a":["b"]}}"#),
            ("dependentSchemas", r#"{"type":"object","dependentSchemas":{"a":{}}}"#),
            ("dependentRequired", r#"{"type":"object","dependentRequired":{"a":["b"]}}"#),
            ("unevaluatedProperties", r#"{"type":"object","unevaluatedProperties":false}"#),
            ("unevaluatedItems", r#"{"type":"array","unevaluatedItems":false}"#),
            // unknown keywords are refused just as loudly
            ("title", r#"{"type":"string","title":"nope"}"#),
            ("x-vendor", r#"{"type":"string","x-vendor":true}"#),
        ];
        for (kw, js) in cases {
            let e = compile_err(js);
            assert!(e.contains(&format!("'{kw}'")), "error must name '{kw}': {e}");
            assert!(e.contains("not supported"), "error for '{kw}' must say so: {e}");
        }
        // ... and the message tells the caller what to do instead.
        let e = compile_err(r#"{"type":"string","pattern":"^a$"}"#);
        assert!(e.contains("V1 supports"), "must list the supported keywords: {e}");
    }

    #[test]
    fn type_dependent_keywords_are_named_loudly() {
        // A supported keyword on the wrong type is refused by name, never ignored.
        for (kw, js) in [
            ("minLength", r#"{"minLength":3}"#),
            ("minItems", r#"{"type":"string","minItems":1}"#),
            ("minimum", r#"{"type":"string","minimum":1}"#),
            ("properties", r#"{"type":"array","properties":{"a":{"type":"string"}}}"#),
            ("enum", r#"{"type":"boolean","enum":["a"]}"#),
        ] {
            let e = compile_err(js);
            assert!(e.contains(&format!("'{kw}'")), "error must name '{kw}': {e}");
        }
        let e = compile_err(r#"{"type":"union_thing"}"#);
        assert!(e.contains("'type'"), "{e}");
        let e = compile_err(r#"{"type":["string","null"]}"#);
        assert!(e.contains("'type'"), "{e}");
    }

    // ---- 2: basic object ---------------------------------------------------------------------

    #[test]
    fn object_required_and_additional_properties() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#);
        assert!(ok(&m, r#"{"a":"x"}"#));
        assert!(ok(&m, r#"{ "a" : "x" }"#)); // whitespace is free
        assert!(ok(&m, r#"{"a":"\u0078"}"#)); // escapes decode to the same string
        assert!(!ok(&m, r#"{"a":1}"#), "string property must reject a number");
        assert!(!ok(&m, r#"{}"#), "missing required key");
        assert!(!ok(&m, r#"{"a":"x","b":1}"#), "additionalProperties defaults to false");
        assert!(!ok(&m, r#"{"a":"x"}extra"#), "trailing junk after the document");
        assert!(!ok(&m, r#"{"a":"x","a":"x"}"#), "duplicate key");
        assert!(!ok(&m, r#"{"a":"x",}"#), "trailing comma in object");
        assert!(!ok(&m, r#"{"a":"x""#), "unterminated document is not complete");
        assert!(!ok(&m, r#"{"b":"x"}"#), "unknown key with additionalProperties false");
        assert!(!ok(&m, r#"[]"#), "wrong value kind");

        // explicit additionalProperties: true re-opens the key set (each at most once)
        let open = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":true}"#);
        assert!(ok(&open, r#"{"a":"x","b":1}"#));
        assert!(ok(&open, r#"{"b":null,"a":"x"}"#));
        assert!(!ok(&open, r#"{"a":"x","a":"y"}"#), "duplicates are rejected even for declared keys");
        assert!(!ok(&open, r#"{"b":1,"b":2}"#), "duplicates are rejected for extra keys too");
        assert!(!ok(&open, r#"{"b":1}"#), "required still applies");

        // a bare object / json_object accepts any shape of object
        let any_obj = mask(r#"{"type":"json_object"}"#);
        assert!(ok(&any_obj, r#"{}"#));
        assert!(ok(&any_obj, r#"{"anything":[1,{"x":null}]}"#));
        assert!(!ok(&any_obj, r#"[]"#));
        assert!(ok(&mask(r#"{"type":"object"}"#), r#"{"z":1}"#));
    }

    // ---- 3: enum -----------------------------------------------------------------------------

    #[test]
    fn enum_prefix_rule() {
        let m = mask(r#"{"enum":["alpha","beta"]}"#);
        assert!(ok(&m, r#""alpha""#));
        assert!(ok(&m, r#""beta""#));
        assert!(!ok(&m, r#""gamma""#), "must die as soon as the prefix leaves the enum");
        assert!(!ok(&m, r#""alph""#), "an unterminated string is not a complete value");
        // the prefix rule is visible at the byte level: after `"` only a/b may start
        let st = walk(&m, r#"""#).unwrap();
        assert!(m.step(st, b'a' as u32).is_some());
        assert!(m.step(st, b'b' as u32).is_some());
        assert!(m.step(st, b'g' as u32).is_none());

        let typed = mask(r#"{"type":"string","enum":["a","b"]}"#);
        assert!(ok(&typed, r#""a""#));
        assert!(!ok(&typed, r#""c""#));
        assert!(!ok(&typed, r#"1"#), "declared type still applies");

        let ints = mask(r#"{"type":"integer","enum":[1,2]}"#);
        assert!(ok(&ints, "1"));
        assert!(ok(&ints, "2"));
        assert!(!ok(&ints, "3"));
        assert!(!ok(&ints, "12"));
        assert!(!ok(&ints, "1.0"), "integer forbids a fraction");
        assert!(!ok(&ints, r#""1""#));
    }

    // ---- 4: array bounds ---------------------------------------------------------------------

    #[test]
    fn array_item_bounds() {
        let m = mask(r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"#);
        assert!(!ok(&m, "[]"), "minItems 1");
        assert!(ok(&m, "[1]"));
        assert!(ok(&m, "[1,2]"));
        assert!(ok(&m, "[ 1 , 2 ]"));
        assert!(!ok(&m, "[1,2,3]"), "maxItems 2");
        assert!(!ok(&m, "[[1]]"), "items must be integers");
        assert!(!ok(&m, "[1,]"), "trailing comma");
        assert!(!ok(&m, "[1")); // unterminated
        let unbounded = mask(r#"{"type":"array"}"#);
        assert!(ok(&unbounded, "[]"));
        assert!(ok(&unbounded, r#"[1,"a",null,{"b":[]}]"#));
    }

    // ---- 5: nesting --------------------------------------------------------------------------

    #[test]
    fn nested_object_array_object() {
        let m = mask(
            r#"{"type":"object","properties":{"xs":{"type":"array","items":{"type":"object",
               "properties":{"n":{"type":"integer","minimum":0}},"required":["n"],
               "additionalProperties":false}}},"required":["xs"],"additionalProperties":false}"#,
        );
        assert!(ok(&m, r#"{"xs":[{"n":1},{"n":2}]}"#));
        assert!(ok(&m, r#"{"xs":[]}"#));
        assert!(!ok(&m, r#"{"xs":[{"n":-1}]}"#), "minimum applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"n":1,"z":0}]}"#), "additionalProperties applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"z":1}]}"#), "required applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"n":1}]"#), "unterminated nesting");
        assert!(!ok(&m, r#"{"xs":{"n":1}}"#), "object where the array belongs");
        assert!(m.summary().contains("root=object"), "{}", m.summary());
        assert!(m.summary().contains("max_depth=4"), "{}", m.summary());
    }

    #[test]
    fn depth_cap_is_loud() {
        let deep = |levels: usize| {
            let mut v = serde_json::json!({"type":"string"});
            for _ in 0..levels {
                v = serde_json::json!({"type":"object","properties":{"x": v}});
            }
            v.to_string()
        };
        // 12 nested property levels are fine; 13 are refused by name.
        assert!(compile_response_format(&serde_json::json!({
            "type":"json_schema",
            "json_schema":{"name":"deep","schema": serde_json::from_str::<Value>(&deep(11)).unwrap()}
        }))
        .is_ok());
        let v: Value = serde_json::from_str(&deep(13)).unwrap();
        let rf = serde_json::json!({"type":"json_schema","json_schema":{"name":"deep","schema": v}});
        let e = compile_response_format(&rf).expect_err("13 levels must be refused");
        assert!(e.contains("'properties'"), "{e}");
        assert!(e.contains("deeper than the V1 limit"), "{e}");
    }

    // ---- 6: token-level layer ----------------------------------------------------------------

    #[test]
    fn tiny_vocab_allowed_and_step_agree() {
        let mut m = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#);
        // Pieces exercise the trie: single bytes, mid-document fragments, and whole documents.
        let pieces: Vec<(u32, Vec<u8>)> = vec![
            (1, b"{".to_vec()),
            (2, b"\"a\"".to_vec()),
            (3, b"}".to_vec()),
            (4, b"1".to_vec()),
            (5, b"\"x\"".to_vec()),
            (6, b":".to_vec()),
            (7, b",".to_vec()),
            (8, b"{\"".to_vec()),
            (9, b"a\":".to_vec()),
            (10, b"\"x\"}".to_vec()),
            (11, b" ".to_vec()),
            (12, b"{\"a\":\"x\"}".to_vec()),
        ];
        m.set_vocab(pieces);
        let words = vocab_words(13);
        assert_eq!(words, 1);

        let s0 = m.start_state();
        // At the start only tokens whose whole byte run is accepted may fire: `{`, `{"`, a space
        // (leading whitespace is legal) and the complete document.
        assert_eq!(m.allowed(s0).as_slice(), bits(&[1, 8, 11, 12], words).as_slice());
        for bad in [2u32, 3, 4, 5, 6, 7, 9, 10] {
            assert_eq!(m.step(s0, bad), None, "token {bad} must not be usable at the start");
        }

        // Walk a full document with the tokenizer's pieces, asserting each use was allowed.
        let used = [1u32, 2, 6, 5, 3];
        let mut st = s0;
        let mut produced = String::new();
        for tok in used {
            let allowed = m.allowed(st);
            assert!(allowed[(tok >> 5) as usize] & (1u32 << (tok & 31)) != 0, "token {tok} was not allowed");
            st = m.step(st, tok).expect("used token must be steppable");
            produced.push_str(match tok {
                1 => "{",
                2 => "\"a\"",
                6 => ":",
                5 => "\"x\"",
                3 => "}",
                _ => unreachable!(),
            });
        }
        assert_eq!(produced, r#"{"a":"x"}"#);
        assert!(m.is_complete(st), "document finished");

        // State-specific expectations (also shows required-key handling inside `allowed`).
        let after_brace = m.step(s0, 1).unwrap();
        assert_eq!(m.allowed(after_brace).as_slice(), bits(&[2, 11], words).as_slice());
        assert_eq!(m.step(after_brace, 3), None, "cannot close before 'a' is present");

        let after_key = m.step(after_brace, 2).unwrap();
        assert_eq!(m.allowed(after_key).as_slice(), bits(&[6, 11], words).as_slice());

        let after_colon = m.step(after_key, 6).unwrap();
        assert_eq!(m.allowed(after_colon).as_slice(), bits(&[2, 5, 10, 11], words).as_slice());
        assert_eq!(m.step(after_colon, 4), None, "1 is not a string");

        let after_value = m.step(after_colon, 5).unwrap();
        assert_eq!(m.allowed(after_value).as_slice(), bits(&[3, 7, 11], words).as_slice());

        let done = m.step(after_value, 3).unwrap();
        assert!(m.is_complete(done));
        // Trailing whitespace keeps the machine in the same (interned) completed state.
        assert_eq!(m.step(done, 11), Some(done));
        assert_eq!(m.allowed(done).as_slice(), bits(&[11], words).as_slice());

        // End-of-generation tokens are only ever reachable through end_mask.
        let em = m.end_mask(&[3]);
        assert_eq!(em.as_slice(), bits(&[3], words).as_slice());
        assert_eq!(m.allowed(done).as_slice(), bits(&[11], words).as_slice());
    }

    #[test]
    fn unknown_and_empty_tokens_are_never_allowed() {
        let mut m = mask(r#"{"type":"integer"}"#);
        m.set_vocab(vec![(0, Vec::new()), (1, b"4".to_vec()), (2, b"2".to_vec())]);
        let s0 = m.start_state();
        let a = m.allowed(s0);
        assert_eq!(a.as_slice(), bits(&[1, 2], 1).as_slice());
        assert_eq!(m.step(s0, 0), None, "empty piece cannot advance the machine");
        assert_eq!(m.step(s0, 7), None, "token missing from the table");
        assert_eq!(m.step(s0, 9999), None);
    }

    #[test]
    fn vocab_not_set_fails_closed() {
        // A mask with no vocabulary attached must emit nothing, not everything.
        let bare = compile_response_format(&serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": {"type": "string"}}
        }))
        .unwrap()
        .expect("schema is enforced");
        assert!(!bare.is_dead(bare.start_state()), "the machine itself is alive");
        assert!(bare.allowed(bare.start_state()).is_empty(), "no vocab: nothing may be emitted");
        // ... and an explicitly empty vocabulary behaves the same way.
        let mut empty = mask(r#"{"type":"string"}"#);
        empty.set_vocab(Vec::new());
        assert!(empty.allowed(empty.start_state()).is_empty());
    }

    // ---- 7: completeness ---------------------------------------------------------------------

    #[test]
    fn complete_state_tracks_document_end() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#);
        let s0 = m.start_state();
        assert!(!m.is_complete(s0), "empty output is not a document");
        for prefix in [r#"{"#, r#"{"a"#, r#"{"a":"#, "{\"a\":1"] {
            let st = walk(&m, prefix).unwrap_or_else(|| panic!("prefix {prefix} should be steppable"));
            assert!(!m.is_complete(st), "{prefix} is not finished");
        }
        let mid = walk(&m, r#"{"a":1"#).unwrap();
        assert!(!m.is_complete(mid), "the object is still open");
        // A pending number IS finished at the top level: `is_complete` finalizes it.
        let nums = mask(r#"{"type":"integer","maximum":10}"#);
        assert!(nums.is_complete(walk(&nums, "7").unwrap()));
        assert!(!nums.is_complete(walk(&nums, "70").unwrap()), "over the bound");
        let finished = walk(&m, r#"{"a":1}"#).unwrap();
        assert!(m.is_complete(finished));
        assert!(m.is_complete(walk(&m, "{\"a\":1} \n\t").unwrap()), "trailing whitespace");
        assert!(walk(&m, r#"{"a":1} x"#).is_none(), "junk is refused outright");
        assert!(walk(&m, r#"{"a":false}"#).is_none(), "wrong value kind is refused");
        assert!(!m.is_dead(finished), "a finished document is alive");
        assert!(m.is_dead(12345), "unknown states report dead (fail closed)");
    }

    // ---- 8: cache ----------------------------------------------------------------------------

    #[test]
    fn allowed_is_cached_and_idempotent() {
        let m = mask(r#"{"type":"integer","minimum":5,"maximum":10}"#);
        let s0 = m.start_state();
        let a = m.allowed(s0);
        let b = m.allowed(s0);
        assert_eq!(a, b, "same bitset");
        assert!(Arc::ptr_eq(&a, &b), "second call must be the cached Arc");
        let before = m.summary();
        assert!(before.contains("cached_masks=1"), "{before}");
        assert!(before.contains("vocab=256 tokens/8 words"), "{before}");
        let _ = m.allowed(s0);
        assert!(m.summary().contains("cached_masks=1"), "no growth on repeat");
    }

    // ---- additional coverage: numbers, strings, response_format shapes ------------------------

    #[test]
    fn number_grammar_and_bounds() {
        let m = mask(r#"{"type":"integer","minimum":5,"maximum":10}"#);
        assert!(ok(&m, "5"));
        assert!(ok(&m, "10"));
        assert!(!ok(&m, "4"));
        assert!(!ok(&m, "11"));
        assert!(!ok(&m, "05"), "leading zero");
        assert!(!ok(&m, "-5"));
        assert!(!ok(&m, "5.0"), "integer forbids fraction");
        assert!(!ok(&m, "1e1"), "integer forbids exponent");
        assert!(!ok(&m, "5x"));
        assert!(!ok(&m, "-"));

        let n = mask(r#"{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1}"#);
        assert!(ok(&n, "0.5"));
        assert!(ok(&n, "1e-3"));
        assert!(!ok(&n, "0"), "exclusive bound");
        assert!(!ok(&n, "1"), "exclusive bound");
        assert!(!ok(&n, "-1"));
        assert!(!ok(&n, "1."), "dangling fraction point");
        assert!(!ok(&n, "1e"), "dangling exponent");

        let e = mask(r#"{"type":"number","maximum":2,"exclusiveMaximum":2}"#);
        assert!(!ok(&e, "2"));
    }

    #[test]
    fn string_lengths_are_code_points() {
        let m = mask(r#"{"type":"string","minLength":2,"maxLength":3}"#);
        assert!(ok(&m, r#""ab""#));
        assert!(ok(&m, r#""abc""#));
        assert!(!ok(&m, r#""a""#));
        assert!(!ok(&m, r#""abcd""#));
        assert!(ok(&m, r#""\u0041B""#), "an escape is one code point");
        assert!(ok(&m, r#""\ud83d\ude00x""#), "a surrogate pair is one code point");
        assert!(!ok(&m, r#""\ud83d\ude00""#), "one code point < minLength 2");
        let one = mask(r#"{"type":"string","maxLength":1}"#);
        assert!(ok(&one, "\"\u{1F600}\""), "4 UTF-8 bytes, 1 code point");
        assert!(!ok(&one, r#""ab""#));
        assert!(!ok(&m, "\"a\nb\""), "raw newline must be escaped");
        assert!(ok(&m, r#""a\nb""#), "escaped newline is fine");
        assert!(!ok(&m, r#""\ud83d""#), "lone high surrogate");
        assert!(!ok(&m, r#""\q""#), "unknown escape");
    }

    #[test]
    fn response_format_shapes() {
        // nothing to enforce
        assert!(compile_response_format(&Value::Null).unwrap().is_none());
        assert!(compile_response_format(&serde_json::json!({"type":"text"})).unwrap().is_none());
        // json_object = any object
        let m = compile_response_format(&serde_json::json!({"type":"json_object"}))
            .unwrap()
            .expect("json_object is enforced");
        assert!(m.summary().contains("root=object"), "{}", m.summary());
        // bare form: {"type":"json_schema","schema":{...}}
        let bare = compile_response_format(
            &serde_json::json!({"type":"json_schema","schema":{"type":"string"}}),
        )
        .unwrap()
        .expect("bare form is enforced");
        assert!(bare.summary().contains("root=string"), "{}", bare.summary());
        // strict/name/description are accepted and ignored
        assert!(compile_response_format(&serde_json::json!({
            "type":"json_schema",
            "json_schema":{"name":"n","description":"d","strict":true,"schema":{"type":"null"}}
        }))
        .is_ok());
        // loud failures
        let e = compile_response_format(&serde_json::json!({
            "type":"json_schema","json_schema":{"name":"n"}
        }))
        .expect_err("missing schema");
        assert!(e.contains("'schema'"), "{e}");
        let e = compile_response_format(&serde_json::json!({"type":"yaml"})).expect_err("bad type");
        assert!(e.contains("'type'"), "{e}");
        let e = compile_response_format(&serde_json::json!({"nope":1})).expect_err("no type");
        assert!(e.contains("'type'"), "{e}");
        assert!(compile_response_format(&serde_json::json!([1,2])).is_err());
        assert!(compile_response_format(&serde_json::json!({"type":"json_schema","json_schema":{"name":"n","schema":{"type":"string"}},"extra":1})).is_err());
    }

    #[test]
    fn permissive_and_literal_values() {
        let m = mask("{}");
        assert!(ok(&m, "123"));
        assert!(ok(&m, "-1.5e3"));
        assert!(ok(&m, r#""s""#));
        assert!(ok(&m, "true"));
        assert!(ok(&m, "null"));
        assert!(ok(&m, r#"[1,"a"]"#));
        assert!(!ok(&m, "tru"));
        assert!(!ok(&m, "[1,]"));
        assert!(ok(&mask(r#"{"type":"boolean"}"#), "false"));
        assert!(!ok(&mask(r#"{"type":"boolean"}"#), "truex"));
        assert!(ok(&mask(r#"{"type":"null"}"#), "null"));
        assert!(!ok(&mask(r#"{"type":"null"}"#), "nul"));
    }
}
