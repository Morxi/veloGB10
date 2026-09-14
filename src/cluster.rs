//! `--head` / `--node` cluster orchestration for 2-node TP=2 (Stage 3).
//!
//! Goal (user): launch one binary per box, zero manual config or model copy. The **head** owns the
//! model; it auto-discovers **node**s, checks binary compatibility, and ships whatever artifacts the
//! node is missing. A **content-addressed cache** means a node that already has the files (or a re-run)
//! transfers nothing.
//!
//! Control plane = normal network (UDP discovery + TCP sync). RDMA (`net.rs`) is reserved for the
//! inference data plane only — bootstrap never depends on verbs, so recovery stays simple.
//!
//! MVP scope: whole-model distribution (per-rank shard distribution is a later optimization once the
//! G-D weight sharding exists). After sync the node has an assembled model dir ready for the TP run.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket, SocketAddr, IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const PROTOCOL_VERSION: u32 = 5;   // v5: + DraftManifest/DraftReady — the DFlash2 drafter rides the sync
                                     //     into the node's blob cache (v4: per-epoch payload lengths)
const DISCOVERY_PORT: u16 = 29499;          // UDP; TCP control plane defaults to 29500 (--port)
const DISCOVERY_MAGIC: &str = "GB10-TP-DISCOVER";
/// Binary-compat token: same compiled kernels + same Rust-side sources + protocol => same wire
/// behavior. Cheaper than hashing the 15 MB executable each launch, and it is exactly what must
/// match across boxes. `-k` covers the kernels (KERNEL_BUILD_ID), `-r` the Rust sources + C shim
/// (SOURCE_BUILD_ID): the sharders/protocol/scheduler change behavior without touching a .cu.
fn binary_version() -> String {
    format!("v{}-k{}-r{}", PROTOCOL_VERSION, env!("KERNEL_BUILD_ID"), env!("SOURCE_BUILD_ID"))
}

// ---------------------------------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Artifact {
    pub logical: String,   // path relative to the model dir, e.g. "config.json"
    pub hash: String,      // sha256 hex
    pub size: u64,
}

#[derive(Serialize, Deserialize, Debug)]
enum Msg {
    Hello { version: String, role: String, hostname: String },
    Manifest { model_id: String, artifacts: Vec<Artifact> },
    Missing { hashes: Vec<String> },
    BlobHeader { hash: String, size: u64 },
    Ready { model_dir: String },
    Config(crate::tp::TpConfig),
    /// v5 — the DFlash2 draft artifact round, ALWAYS sent right after `Config` (empty artifacts
    /// = "no draft this session") so both sides stay in strict lockstep without the node having
    /// to guess from its config. The node answers `Missing` (shared with the model round), the
    /// head streams `BlobHeader`+bytes, and the node answers `DraftReady` with the assembled
    /// cache path. Nodes therefore NEVER need a local draft copy (owner gap fix 2026-08-23).
    DraftManifest { model_id: String, artifacts: Vec<Artifact> },
    DraftReady { model_dir: String },
    Error { msg: String },
}

/// Length-prefixed JSON framing, shared by the sync protocol (`Msg`) and, once a session is
/// retained, by the TP serving control plane (`tp_serve::ServingMsg`).
pub(crate) fn send_json<T: Serialize>(w: &mut impl Write, m: &T) -> Result<()> {
    let b = serde_json::to_vec(m)?;
    w.write_all(&(b.len() as u32).to_be_bytes())?;
    w.write_all(&b)?;
    w.flush()?;
    Ok(())
}
pub(crate) fn recv_json<T: serde::de::DeserializeOwned>(r: &mut impl Read) -> Result<T> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let n = u32::from_be_bytes(len) as usize;
    if n > 64 * 1024 * 1024 { bail!("control message too large ({n} B)"); }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(serde_json::from_slice(&b)?)
}

fn send_msg(w: &mut impl Write, m: &Msg) -> Result<()> { send_json(w, m) }
fn recv_msg(r: &mut impl Read) -> Result<Msg> { recv_json(r) }

// ---------------------------------------------------------------------------------------------------
// Content-addressed cache
// ---------------------------------------------------------------------------------------------------

fn cache_root() -> PathBuf {
    std::env::var("GB10_TP_CACHE").map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".cache/gb10_tp")
    })
}
fn blob_path(hash: &str) -> PathBuf { cache_root().join("blobs").join(hash) }
fn have_blob(hash: &str) -> bool { blob_path(hash).exists() }

/// Atomically publish `tmp` (already hash-verified) into the content store as `blobs/<hash>`.
fn publish_blob(hash: &str, tmp: &Path) -> Result<()> {
    let dst = blob_path(hash);
    std::fs::create_dir_all(dst.parent().unwrap())?;
    std::fs::rename(tmp, &dst).with_context(|| format!("publish blob {hash}"))?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex(&h.finalize()), total))
}
fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

/// Per-file (path, mtime, size) -> hash cache so re-launches don't re-hash a 15 GB model.
fn hash_cache_load() -> HashMap<String, String> {
    let p = cache_root().join("hashcache.json");
    std::fs::read(&p).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}
fn hash_cache_save(m: &HashMap<String, String>) {
    let p = cache_root().join("hashcache.json");
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    if let Ok(b) = serde_json::to_vec(m) { let _ = std::fs::write(&p, b); }
}
fn cached_key(path: &Path) -> Result<String> {
    let md = std::fs::metadata(path)?;
    let mtime = md.modified()?.duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    Ok(format!("{}|{}|{}", path.display(), mtime, md.len()))
}

// ---------------------------------------------------------------------------------------------------
// Manifest (head side)
// ---------------------------------------------------------------------------------------------------

/// Files a node needs to serve a model. Follows a symlinked model dir to the real files.
fn model_files(dir: &Path) -> Result<Vec<PathBuf>> {
    // Recursive walk: a bundle may carry required subdirs the loader reads (DSV4's
    // `inference/config.json`, `encoding/`). Top-level-only shipping starves the node of those and
    // it crashes at load. Symlinks followed; editor/OS cruft + our sidecars skipped at every level.
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).with_context(|| format!("read_dir {}", d.display()))? {
            let e = entry?;
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name.ends_with(".tmp") { continue; }
            let meta = std::fs::metadata(&p)?; // follows symlinks
            if meta.is_file() {
                out.push(p);
            } else if meta.is_dir() {
                stack.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

pub fn build_manifest(model_dir: &Path, world: u32) -> Result<(String, Vec<Vec<Artifact>>)> {
    let world = world.max(1) as usize;
    let mut cache = hash_cache_load();
    let mut dirty = false;
    // Per-rank artifact lists indexed by rank. rank 0 is the head's own shard and is never shipped;
    // ranks 1..world-1 are each node's manifest. When `rank{r}/` exists the head ships ONLY that
    // node's shard (+ the root inference/config.json + root config.json, exactly the world==2 set);
    // when a rank dir does NOT exist, that rank receives the whole model (replicated, the P2
    // replicate-if-not-divisible rule). For world==2 this is byte-identical to the pre-P4 single
    // `rank1/` manifest.
    let mut per_rank: Vec<Vec<Artifact>> = vec![Vec::new(); world];
    let sharded: Vec<bool> = (1..world)
        .map(|r| model_dir.join(format!("rank{r}")).exists())
        .collect();
    for path in model_files(model_dir)? {
        let rel = path.strip_prefix(model_dir).unwrap_or(&path).to_string_lossy().to_string();
        let key = cached_key(&path)?;
        let (hash, size) = if let Some(h) = cache.get(&key) {
            (h.clone(), std::fs::metadata(&path)?.len())
        } else {
            let (h, sz) = sha256_file(&path)?;
            cache.insert(key, h.clone());
            dirty = true;
            (h, sz)
        };
        let artifact = Artifact { logical: rel.clone(), hash, size };
        for r in 1..world {
            if include_for_rank(&rel, r, sharded[r - 1]) {
                per_rank[r].push(artifact.clone());
            }
        }
    }
    if dirty { hash_cache_save(&cache); }
    let model_id = model_dir.file_name().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());
    Ok((model_id, per_rank))
}

/// Whether a logical path (relative to the model dir) belongs in node rank `r`'s manifest. When
/// `rank{r}/` exists (`sharded`), only files under `rank{r}/` plus the two always-shipped root
/// files are kept; otherwise the whole model is replicated (P2 replicate-if-not-divisible).
fn include_for_rank(rel: &str, rank: usize, sharded: bool) -> bool {
    if !sharded {
        return true;
    }
    let rank_dir = format!("rank{rank}");
    rel.starts_with(&format!("{rank_dir}/")) || rel == "inference/config.json" || rel == "config.json"
}

/// Manifest for a REPLICATED auxiliary artifact dir — the DFlash2 drafter (gap fix 2026-08-23).
/// Unlike `build_manifest` there is no rank partitioning: every file goes to every node (the
/// drafter is never sharded over the sync; `Df2Round::load_tp` shards it in memory after load).
/// Shares the per-path (path,mtime,size)→hash cache with the trunk manifest, so a re-launch
/// re-hashes nothing.
fn draft_manifest(dir: &Path) -> Result<(String, Vec<Artifact>)> {
    let mut cache = hash_cache_load();
    let mut dirty = false;
    let mut artifacts = Vec::new();
    for path in model_files(dir)? {
        let rel = path.strip_prefix(dir).unwrap_or(&path).to_string_lossy().to_string();
        let key = cached_key(&path)?;
        let (hash, size) = if let Some(h) = cache.get(&key) {
            (h.clone(), std::fs::metadata(&path)?.len())
        } else {
            let (h, sz) = sha256_file(&path)?;
            cache.insert(key, h.clone());
            dirty = true;
            (h, sz)
        };
        artifacts.push(Artifact { logical: rel, hash, size });
    }
    if dirty { hash_cache_save(&cache); }
    let model_id = dir.file_name().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "draft".into());
    Ok((model_id, artifacts))
}

// ---------------------------------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NodeInfo { pub hostname: String, pub addr: SocketAddr }

#[derive(Serialize, Deserialize)]
struct DiscoverProbe { magic: String, version: String }
#[derive(Serialize, Deserialize)]
struct DiscoverReply { hostname: String, tcp_port: u16, version: String }

/// Node side: answer discovery probes with our on-path IP + TCP port. Runs until the process exits.
pub fn spawn_discovery_responder(tcp_port: u16) -> Result<()> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, DISCOVERY_PORT))
        .context("bind UDP discovery port")?;
    let hostname = hostname();
    std::thread::spawn(move || {
        let mut buf = [0u8; 2048];
        loop {
            let (n, from) = match sock.recv_from(&mut buf) { Ok(x) => x, Err(_) => continue };
            let probe: DiscoverProbe = match serde_json::from_slice(&buf[..n]) { Ok(p) => p, Err(_) => continue };
            if probe.magic != DISCOVERY_MAGIC { continue; }
            // Reply via the same socket: the OS picks the source IP by the route back to the head, so
            // the head sees our on-path (RoCE) IP as the datagram source — exactly the TCP address to use.
            let reply = DiscoverReply { hostname: hostname.clone(), tcp_port, version: binary_version() };
            if let Ok(b) = serde_json::to_vec(&reply) { let _ = sock.send_to(&b, from); }
        }
    });
    Ok(())
}

/// Head side: broadcast a probe on the RoCE subnets (+ global broadcast) and collect responders.
pub fn discover_nodes(wait: Duration) -> Result<Vec<NodeInfo>> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
    sock.set_broadcast(true)?;
    sock.set_read_timeout(Some(Duration::from_millis(200)))?;
    let probe = serde_json::to_vec(&DiscoverProbe {
        magic: DISCOVERY_MAGIC.into(), version: binary_version() })?;
    // GB10 boxes always expose the same RoCE interface NAMES; resolve their live IP/broadcast at
    // runtime (IPs are config, names/types are fixed) and broadcast there + a global catch-all.
    let ifaces = roce_interfaces();
    for f in &ifaces {
        eprintln!("  [discover] RoCE rail {} {} ({}) ip {} bcast {}", f.rail, f.ib, f.netdev, f.ip, f.bcast);
    }
    if ifaces.is_empty() {
        eprintln!("  [discover] no RoCE interface resolved — broadcasting globally only");
    }
    let mut targets: Vec<Ipv4Addr> = ifaces.iter().map(|f| f.bcast).collect();
    targets.push(Ipv4Addr::BROADCAST);
    for t in &targets { let _ = sock.send_to(&probe, (*t, DISCOVERY_PORT)); }
    let deadline = std::time::Instant::now() + wait;
    // A node replies once per subnet the probe reached it on (RoCE rail 1/2 + mgmt), all same hostname.
    // Keep the RoCE-preferred source IP: that is exactly the address the RDMA data plane must use.
    let mut nodes: HashMap<String, (u8, NodeInfo)> = HashMap::new();
    let mut buf = [0u8; 2048];
    while std::time::Instant::now() < deadline {
        match sock.recv_from(&mut buf) {
            Ok((n, from)) => {
                if let Ok(r) = serde_json::from_slice::<DiscoverReply>(&buf[..n]) {
                    if r.version != binary_version() {
                        eprintln!("  [discover] {} at {} has MISMATCHED binary ({} vs {}) — skipping",
                                  r.hostname, from.ip(), r.version, binary_version());
                        continue;
                    }
                    let rank = ip_rank(from.ip(), &ifaces);
                    let ni = NodeInfo { hostname: r.hostname.clone(), addr: SocketAddr::new(from.ip(), r.tcp_port) };
                    match nodes.get(&r.hostname) {
                        Some((existing, _)) if *existing >= rank => {}
                        _ => { nodes.insert(r.hostname, (rank, ni)); }
                    }
                }
            }
            Err(_) => {}   // read timeout; keep polling until the deadline
        }
    }
    Ok(nodes.into_values().map(|(_, ni)| ni).collect())
}

/// A live ConnectX-7 RoCE interface, resolved from its (fixed) IB device name to its current IPv4.
struct Roce { ib: String, netdev: String, ip: Ipv4Addr, bcast: Ipv4Addr, mask: u32, rail: u8 }

/// Resolve the GB10 RoCE rails by their fixed IB device names → netdev (via /sys) → IPv4 (via `ip`).
/// Names/types are constant across GB10 boxes (per the hardware); only the IPs are configuration.
fn roce_interfaces() -> Vec<Roce> {
    // Default = the fixed GB10 rail names (identical across DGX Spark + every OEM clone: same SoC,
    // hard-wired PCIe topology, systemd predictable naming). Manual fallback for any platform that
    // breaks that: GB10_RDMA_DEV=dev1[,dev2] (rail order), set via --rdma-dev.
    let devs: Vec<(String, u8)> = match std::env::var("GB10_RDMA_DEV") {
        Ok(s) if !s.trim().is_empty() =>
            s.split(',').enumerate().map(|(i, d)| (d.trim().to_string(), (i + 1) as u8)).collect(),
        _ => vec![("rocep1s0f1".into(), 1), ("roceP2p1s0f1".into(), 2)],
    };
    let mut out = Vec::new();
    for (ib, rail) in &devs {
        let netdir = format!("/sys/class/infiniband/{ib}/device/net");
        let netdev = std::fs::read_dir(&netdir).ok()
            .and_then(|mut d| d.next()).and_then(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string());
        let Some(netdev) = netdev else {
            eprintln!("  [discover] RoCE device '{ib}' not found — override with --rdma-dev / GB10_RDMA_DEV, \
                       or use --nodes <ip> to skip discovery");
            continue;
        };
        if let Some((ip, prefix, bcast)) = ipv4_of(&netdev) {
            let mask = if prefix == 0 { 0 } else { u32::MAX << (32 - prefix) };
            out.push(Roce { ib: ib.clone(), netdev, ip, bcast, mask, rail: *rail });
        }
    }
    out
}

/// Parse `ip -o -4 addr show dev <netdev>` → (addr, prefix_len, broadcast).
fn ipv4_of(netdev: &str) -> Option<(Ipv4Addr, u8, Ipv4Addr)> {
    let out = std::process::Command::new("ip")
        .args(["-o", "-4", "addr", "show", "dev", netdev]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let toks: Vec<&str> = s.split_whitespace().collect();
    let (mut ip, mut prefix, mut brd) = (None, None, None);
    let mut i = 0;
    while i + 1 < toks.len() {
        match toks[i] {
            "inet" => { let mut it = toks[i + 1].split('/');
                ip = it.next().and_then(|x| x.parse().ok());
                prefix = it.next().and_then(|x| x.parse().ok()); }
            "brd" => brd = toks[i + 1].parse().ok(),
            _ => {}
        }
        i += 1;
    }
    Some((ip?, prefix?, brd?))
}

/// Rank a reply's source IP: on RoCE rail 1 (3) > rail 2 (2) > any other link (1).
fn ip_rank(ip: std::net::IpAddr, ifaces: &[Roce]) -> u8 {
    if let std::net::IpAddr::V4(v4) = ip {
        let x = u32::from(v4);
        for f in ifaces {
            if x & f.mask == u32::from(f.ip) & f.mask {
                return if f.rail == 1 { 3 } else { 2 };
            }
        }
    }
    1
}

/// Deterministic total order over a node's `SocketAddr`. Higher RoCE-rail preference sorts FIRST,
/// then ascending IPv4, then ascending port. The IPv4 tie-break is what makes the order a *total*
/// order (hostnames may collide across boxes or be unset in explicit `--nodes` mode).
fn node_order_key(addr: &SocketAddr, ifaces: &[Roce]) -> (u8, u128, u16) {
    let rail = ip_rank(addr.ip(), ifaces);
    let ip_key = match addr.ip() {
        IpAddr::V4(v4) => u32::from(v4) as u128,
        IpAddr::V6(v6) => u128::from(v6),
    };
    // rail desc => invert; ip/port asc.
    (255u8.wrapping_sub(rail), ip_key, addr.port())
}

/// Assign node ranks 1..N-1 in a stable, reproducible order. The head is ALWAYS rank 0; the
/// discovered nodes (sorted by `node_order_key`) become ranks 1,2,... in that order. The sort is
/// decoupled from `discover_nodes` (which only picks the best source IP per hostname) so explicit
/// `--nodes` and discovery take the same path.
fn assign_ranks(nodes: &mut Vec<NodeInfo>, ifaces: &[Roce]) {
    nodes.sort_by(|a, b| node_order_key(&a.addr, ifaces).cmp(&node_order_key(&b.addr, ifaces)));
}

/// Build the full rank→RoCE-IP topology (`Vec<String>` indexed by rank, size `world`). The head is
/// rank 0 at its own RoCE rail-1 IP (or the first resolved RoCE IP); `ranked_nodes` must already be
/// sorted by `assign_ranks` (ranks 1..N-1 in order). `topology[self_rank]` is the rank's own IP and
/// is unused by the N-way transport.
fn build_topology(ranked_nodes: &[NodeInfo], ifaces: &[Roce], world: u32) -> Result<Vec<String>> {
    anyhow::ensure!(
        ranked_nodes.len() as u32 == world.saturating_sub(1),
        "build_topology: got {} nodes for world {world} (need {})",
        ranked_nodes.len(),
        world.saturating_sub(1)
    );
    let mut topo: Vec<String> = vec![String::new(); world as usize];
    // rank 0 = this process; use its own RoCE rail-1 IP (fall back to any resolved RoCE IP). This
    // entry is the address every node uses to dial the control QP, so it MUST be non-empty.
    let head_ip = ifaces.iter().find(|f| f.rail == 1).map(|f| f.ip)
        .or_else(|| ifaces.first().map(|f| f.ip))
        .context("no RoCE interface resolved — cannot build the rank->IP topology")?;
    topo[0] = head_ip.to_string();
    for (i, node) in ranked_nodes.iter().enumerate() {
        let rank = (i + 1) as usize;
        topo[rank] = node.addr.ip().to_string();
    }
    Ok(topo)
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "node".into())
}

// ---------------------------------------------------------------------------------------------------
// Blob streaming
// ---------------------------------------------------------------------------------------------------

fn send_blob(w: &mut impl Write, path: &Path) -> Result<()> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        w.write_all(&buf[..n])?;
    }
    w.flush()?;
    Ok(())
}

/// Receive `size` bytes into a temp file, hashing as we go; verify against `hash`; return the temp path.
fn recv_blob(r: &mut impl Read, hash: &str, size: u64) -> Result<PathBuf> {
    let tmp = cache_root().join("blobs").join(format!("tmp.{}.{}", std::process::id(), hash));
    std::fs::create_dir_all(tmp.parent().unwrap())?;
    let mut f = std::fs::File::create(&tmp)?;
    let mut hasher = Sha256::new();
    let mut left = size;
    let mut buf = vec![0u8; 1 << 20];
    while left > 0 {
        let want = left.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..want])?;
        hasher.update(&buf[..want]);
        f.write_all(&buf[..want])?;
        left -= want as u64;
    }
    f.flush()?;
    let got = hex(&hasher.finalize());
    if got != hash {
        let _ = std::fs::remove_file(&tmp);
        bail!("blob hash mismatch: expected {hash}, got {got}");
    }
    Ok(tmp)
}

// ---------------------------------------------------------------------------------------------------
// Node: receive-and-assemble
// ---------------------------------------------------------------------------------------------------

/// Assemble a model dir under the cache: each logical name -> symlink to blobs/<hash>.
fn assemble_model_dir(model_id: &str, artifacts: &[Artifact]) -> Result<PathBuf> {
    let dir = cache_root().join("models").join(model_id);
    std::fs::create_dir_all(&dir)?;
    for a in artifacts {
        let link = dir.join(&a.logical);
        if let Some(parent) = link.parent() { std::fs::create_dir_all(parent)?; }
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(blob_path(&a.hash), &link)
            .with_context(|| format!("symlink {}", a.logical))?;
    }
    Ok(dir)
}

/// Handle one head connection: Hello -> Manifest -> Missing -> blobs -> assemble -> Ready ->
/// Config -> DraftManifest -> (draft blobs) -> DraftReady. Returns the assembled model dir +
/// the assembled DRAFT cache dir (None when the head shipped no drafter) + the head's TP config
/// (so the node needs ZERO GB10_TP_* env, ZERO model paths, ZERO draft paths) + the RETAINED
/// control stream (the TP serving session runs over it; the bench path just drops it).
fn node_handle(mut s: TcpStream) -> Result<(PathBuf, Option<PathBuf>, crate::tp::TpConfig, TcpStream)> {
    match recv_msg(&mut s)? {
        Msg::Hello { version, hostname, .. } => {
            if version != binary_version() {
                send_msg(&mut s, &Msg::Error { msg: format!("binary mismatch: head {version} vs node {}", binary_version()) })?;
                bail!("head/node binary mismatch");
            }
            eprintln!("  [node] head '{hostname}' connected (binary {version})");
        }
        _ => bail!("expected Hello"),
    }
    send_msg(&mut s, &Msg::Hello { version: binary_version(), role: "node".into(), hostname: hostname() })?;

    let (model_id, artifacts) = match recv_msg(&mut s)? {
        Msg::Manifest { model_id, artifacts } => (model_id, artifacts),
        _ => bail!("expected Manifest"),
    };
    let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
        .filter(|h| !have_blob(h)).collect();
    let have = artifacts.len() - missing.len();
    eprintln!("  [node] manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
              artifacts.len(), missing.len());
    send_msg(&mut s, &Msg::Missing { hashes: missing.clone() })?;

    for _ in 0..missing.len() {
        match recv_msg(&mut s)? {
            Msg::BlobHeader { hash, size } => {
                let tmp = recv_blob(&mut s, &hash, size)?;
                publish_blob(&hash, &tmp)?;
                eprintln!("  [node] cached {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
            }
            _ => bail!("expected BlobHeader"),
        }
    }
    let dir = assemble_model_dir(&model_id, &artifacts)?;
    send_msg(&mut s, &Msg::Ready { model_dir: dir.to_string_lossy().to_string() })?;
    let cfg = match recv_msg(&mut s)? {
        Msg::Config(c) => c,
        _ => bail!("expected Config"),
    };
    eprintln!("  [node] config from head: shard_mixers={} graph={} fp32_partials={} mtp={} depth={:?} mode_serve={}",
              cfg.shard_mixers, cfg.graph, cfg.fp32_partials, cfg.mtp, cfg.mtp_depth, cfg.mode_serve);

    // v5 — the DFlash2 draft round (always exactly one DraftManifest after Config). The bytes
    // land in the SAME content-addressed blob store as the model shards; the assembled dir lives
    // under models/<draft-id>/ exactly like a trunk model. A node therefore NEVER needs a local
    // draft copy — the caller rewrites the config's df2_draft_dir to the returned cache path.
    let draft_dir: Option<PathBuf> = match recv_msg(&mut s)? {
        Msg::DraftManifest { model_id, artifacts } if artifacts.is_empty() => {
            send_msg(&mut s, &Msg::DraftReady { model_dir: String::new() })?;
            eprintln!("  [node] no draft artifact this session (head sent an empty manifest)");
            None
        }
        Msg::DraftManifest { model_id, artifacts } => {
            let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
                .filter(|h| !have_blob(h)).collect();
            let have = artifacts.len() - missing.len();
            eprintln!("  [node] draft manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
                      artifacts.len(), missing.len());
            send_msg(&mut s, &Msg::Missing { hashes: missing.clone() })?;
            for _ in 0..missing.len() {
                match recv_msg(&mut s)? {
                    Msg::BlobHeader { hash, size } => {
                        let tmp = recv_blob(&mut s, &hash, size)?;
                        publish_blob(&hash, &tmp)?;
                        eprintln!("  [node] cached draft blob {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
                    }
                    _ => bail!("expected draft BlobHeader"),
                }
            }
            let ddir = assemble_model_dir(&model_id, &artifacts)?;
            send_msg(&mut s, &Msg::DraftReady { model_dir: ddir.to_string_lossy().to_string() })?;
            eprintln!("  [node] DRAFT READY — drafter assembled at {} (loads from the cache, no local copy)",
                      ddir.display());
            Some(ddir)
        }
        _ => bail!("expected DraftManifest after Config"),
    };
    Ok((dir, draft_dir, cfg, s))
}

/// Run as a node: answer discovery, accept ONE head sync, return the assembled model dir + the head's
/// IP (its RoCE address, used to bring up the RDMA data-plane link back to it) + the head's TP config
/// + the retained control stream (dropped by bench sessions, kept by serving ones).
pub fn run_node(tcp_port: u16)
    -> Result<(PathBuf, Option<PathBuf>, IpAddr, crate::tp::TpConfig, TcpStream)> {
    spawn_discovery_responder(tcp_port)?;
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, tcp_port))
        .with_context(|| format!("bind TCP {tcp_port}"))?;
    eprintln!("[node] {} ready: discovery on UDP {DISCOVERY_PORT}, control on TCP {tcp_port}, cache {}",
              hostname(), cache_root().display());
    let (s, from) = listener.accept()?;
    eprintln!("[node] head connected from {from}");
    s.set_nodelay(true).ok();
    let (dir, draft_dir, cfg, s) = node_handle(s)?;
    eprintln!("[node] SYNCED — model ready at {}{}",
              dir.display(),
              draft_dir.as_ref().map(|d| format!(", drafter at {}", d.display())).unwrap_or_default());
    Ok((dir, draft_dir, from.ip(), cfg, s))
}

// ---------------------------------------------------------------------------------------------------
// Head: discover-and-push
// ---------------------------------------------------------------------------------------------------

/// Push the model to one node: Hello -> Manifest -> receive Missing -> stream blobs -> Ready, then
/// ship our TP config (`Msg::Config`) so the node runs with ZERO GB10_TP_* env vars. The shipped
/// config is a per-node clone: `node_rank` identifies this node and `topology` is the full
/// rank→RoCE-IP map (the node reads both back from its own `Msg::Config`). Returns the RETAINED
/// control stream — the TP serving session keeps it as its control plane; the bench path drops it.
fn head_sync_one(node: &NodeInfo, node_rank: i32, model_dir: &Path, model_id: &str,
                 artifacts: &[Artifact], topology: &[String],
                 cfg: &crate::tp::TpConfig,
                 draft: Option<&(String, Vec<Artifact>)>) -> Result<TcpStream> {
    let mut s = TcpStream::connect_timeout(&node.addr, Duration::from_secs(10))
        .with_context(|| format!("connect {}", node.addr))?;
    s.set_nodelay(true).ok();
    send_msg(&mut s, &Msg::Hello { version: binary_version(), role: "head".into(), hostname: hostname() })?;
    match recv_msg(&mut s)? {
        Msg::Hello { .. } => {}
        Msg::Error { msg } => bail!("node rejected: {msg}"),
        _ => bail!("expected node Hello"),
    }
    send_msg(&mut s, &Msg::Manifest { model_id: model_id.into(), artifacts: artifacts.to_vec() })?;
    let missing = match recv_msg(&mut s)? {
        Msg::Missing { hashes } => hashes,
        _ => bail!("expected Missing"),
    };
    let by_hash: HashMap<&str, &Artifact> = artifacts.iter().map(|a| (a.hash.as_str(), a)).collect();
    let total: u64 = missing.iter().filter_map(|h| by_hash.get(h.as_str())).map(|a| a.size).sum();
    eprintln!("[head] {} (rank {node_rank}) needs {} / {} artifacts ({:.2} GB)", node.hostname,
              missing.len(), artifacts.len(), total as f64 / 1e9);
    let t0 = std::time::Instant::now();
    for h in &missing {
        let a = by_hash.get(h.as_str()).context("node asked for unknown hash")?;
        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
        send_blob(&mut s, &model_dir.join(&a.logical))?;
    }
    match recv_msg(&mut s)? {
        Msg::Ready { model_dir } => {
            let secs = t0.elapsed().as_secs_f64();
            eprintln!("[head] {} (rank {node_rank}) READY — model at {} ({:.2} GB in {:.1}s = {:.2} GB/s)",
                      node.hostname, model_dir, total as f64/1e9, secs,
                      if secs > 0.0 { total as f64/1e9/secs } else { 0.0 });
            let mut node_cfg = cfg.clone();
            node_cfg.node_rank = node_rank;
            node_cfg.topology = topology.to_vec();
            send_msg(&mut s, &Msg::Config(node_cfg.clone()))?;
            eprintln!("[head] shipped config to {} (rank {node_rank}/{})", node.hostname, node_cfg.world);

            // v5 — the DFlash2 draft round: ALWAYS exactly one DraftManifest after Config (empty
            // artifacts = none), so the node's receive state machine is unconditional and can
            // never block waiting for a round the head decided not to send.
            match draft {
                Some((draft_id, draft_arts)) if !draft_arts.is_empty() => {
                    send_msg(&mut s, &Msg::DraftManifest { model_id: draft_id.clone(),
                                                          artifacts: draft_arts.clone() })?;
                    let src_dir = Path::new(&cfg.df2_draft_dir);
                    let dmissing = match recv_msg(&mut s)? {
                        Msg::Missing { hashes } => hashes,
                        _ => bail!("expected draft Missing from {}", node.hostname),
                    };
                    let dby_hash: HashMap<&str, &Artifact> =
                        draft_arts.iter().map(|a| (a.hash.as_str(), a)).collect();
                    let dtotal: u64 = dmissing.iter().filter_map(|h| dby_hash.get(h.as_str())).map(|a| a.size).sum();
                    eprintln!("[head] {} drafter: {} / {} artifacts ({:.2} GB)",
                              node.hostname, dmissing.len(), draft_arts.len(), dtotal as f64 / 1e9);
                    let dt0 = std::time::Instant::now();
                    for h in &dmissing {
                        let a = dby_hash.get(h.as_str()).context("node asked for unknown draft hash")?;
                        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
                        send_blob(&mut s, &src_dir.join(&a.logical))?;
                    }
                    match recv_msg(&mut s)? {
                        Msg::DraftReady { model_dir } => {
                            let secs = dt0.elapsed().as_secs_f64();
                            eprintln!("[head] {} drafter READY at {} ({:.2} GB in {:.1}s)",
                                      node.hostname, model_dir, dtotal as f64/1e9, secs);
                        }
                        Msg::Error { msg } => bail!("node draft error: {msg}"),
                        _ => bail!("expected DraftReady"),
                    }
                }
                _ => {
                    send_msg(&mut s, &Msg::DraftManifest { model_id: String::new(), artifacts: Vec::new() })?;
                    match recv_msg(&mut s)? {
                        Msg::DraftReady { .. } => {}
                        Msg::Error { msg } => bail!("node draft error: {msg}"),
                        _ => bail!("expected DraftReady (none)"),
                    }
                }
            }
            Ok(s)
        }
        Msg::Error { msg } => bail!("node error: {msg}"),
        _ => bail!("expected Ready"),
    }
}

/// Run as the head: discover nodes (or use explicit addrs), assign ranks 1..N-1 deterministically,
/// then sync the model to each node's OWN shard. Sets the process-global topology before returning
/// (so `bring_up_head` can resolve its N-way partners).
pub fn run_head(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                cfg: &crate::tp::TpConfig)
    -> Result<Vec<NodeInfo>>
{
    let world = cfg.world.max(1);
    eprintln!("[head] {} — building manifest for {} (world {world}) ...", hostname(), model_dir.display());
    let (model_id, per_rank) = build_manifest(model_dir, world)?;
    let total: u64 = per_rank.iter().flatten().map(|a| a.size).sum();
    eprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB", total_artifacts(&per_rank), total as f64/1e9);

    let ifaces = roce_interfaces();
    let mut nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            eprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { eprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
            n
        }
    };
    let need = (world - 1) as usize;
    if nodes.len() != need {
        bail!("--tp {world} needs exactly {need} node(s), got {} — start exactly {need} --node (or pass exactly {need} --nodes <ip:port>)",
              nodes.len());
    }
    assign_ranks(&mut nodes, &ifaces);
    let topology = build_topology(&nodes, &ifaces, world)?;
    crate::tp::set_topology(topology.clone());
    for (i, node) in nodes.iter().enumerate() {
        let rank = (i + 1) as i32;
        // Bench sessions ship no drafter (bench nodes never load a round; the bench wire adds
        // only the empty-DraftManifest round to the pre-v5 behavior).
        head_sync_one(node, rank, model_dir, &model_id, &per_rank[rank as usize], &topology, cfg, None)?;
    }
    eprintln!("[head] all {} node(s) synced.", nodes.len());
    Ok(nodes)
}

fn total_artifacts(per_rank: &[Vec<Artifact>]) -> usize {
    per_rank.iter().flatten().count()
}

/// Run as the head for a TP SERVING session (TP item A): identical to `run_head` (same manifest,
/// discovery, and blob push), but the sync connections are RETAINED and returned — the whole serving
/// control plane (calibration table, per-step events, shutdown) then runs over them as
/// `tp_serve::ServingMsg`. The node count requirement is world-aware: exactly `world - 1` nodes.
/// Returns the retained streams RANK-INDEXED (`streams[i]` is the stream to rank `i + 1`), so the
/// head can fan out `CalibTable` / `Step` / `Shutdown` to every node (world > 2) and wait for
/// `Ready` from every node. The node side is `run_node`'s retained stream.
pub fn run_head_session(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                        cfg: &crate::tp::TpConfig)
    -> Result<(Vec<NodeInfo>, Vec<TcpStream>)>
{
    let world = cfg.world.max(1);
    eprintln!("[head] {} — building manifest for {} (world {world}) ...", hostname(), model_dir.display());
    let (model_id, per_rank) = build_manifest(model_dir, world)?;
    let total: u64 = per_rank.iter().flatten().map(|a| a.size).sum();
    eprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB", total_artifacts(&per_rank), total as f64/1e9);

    let ifaces = roce_interfaces();
    let mut nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            eprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { eprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
            n
        }
    };
    let need = (world - 1) as usize;
    if nodes.len() != need {
        bail!("--tp {world} serving needs exactly {need} node(s), got {} — start exactly {need} --node (or pass exactly {need} --nodes <ip:port>)",
              nodes.len());
    }
    assign_ranks(&mut nodes, &ifaces);
    let topology = build_topology(&nodes, &ifaces, world)?;
    crate::tp::set_topology(topology.clone());
    // v5 — the DFlash2 drafter rides the same sync (gap fix 2026-08-23): on serve sessions with
    // a DF2 spec source and a resolved --draft-dir, ship the WHOLE artifact (replicated, content-
    // addressed) so every node loads it from its blob cache instead of a hand-copied local dir.
    // A manifest failure is loud but not fatal here — the head's own round load fails the same
    // way and CalibTable's df2_round=false keeps all ranks consistently on MTP.
    // WI1: the DSpark source ships its artifact through the same content-addressed sync.
    let src_parsed = crate::batch::SpecSource::from_cli(&cfg.spec_source)
        .unwrap_or(crate::batch::SpecSource::Mtp);
    // P14: `dflash` (the v1 BLOCK drafter) ships through the same content-addressed slot: its lane
    // is part of the mirrored `decode_step`, so every rank needs the artifact, and the node must
    // never read the head's filesystem path.
    let draft: Option<(String, Vec<Artifact>)> = if cfg.mode_serve
        && (crate::batch::is_df2_src(src_parsed)
            || matches!(src_parsed, crate::batch::SpecSource::Dspark)
            || matches!(src_parsed, crate::batch::SpecSource::DFlash))
        && !cfg.df2_draft_dir.is_empty()
    {
        match draft_manifest(Path::new(&cfg.df2_draft_dir)) {
            Ok((id, arts)) => {
                let dtotal: u64 = arts.iter().map(|a| a.size).sum();
                eprintln!("[head] draft manifest '{id}': {} artifacts, {:.2} GB — ships to every node",
                          arts.len(), dtotal as f64 / 1e9);
                Some((id, arts))
            }
            Err(e) => {
                eprintln!("[head] WARN: draft manifest build FAILED ({e:#}) — nodes get NO drafter; \
                           the head's round load fails the same way and all ranks fall back to MTP");
                None
            }
        }
    } else { None };

    // Sync every node and RETAIN every node's sync stream, RANK-INDEXED (streams[i] == rank i+1).
    // The serving control plane (CalibTable, per-step Step, Shutdown) must fan out to ALL world-1
    // nodes; dropping rank 2..N-1's streams here is exactly what deadlocked world>2 bring-up (those
    // nodes reached node_serve_tp and blocked forever on a CalibTable the head never sent them).
    let mut streams: Vec<TcpStream> = Vec::with_capacity(need);
    for (i, node) in nodes.iter().enumerate() {
        let rank = (i + 1) as i32;
        let stream = head_sync_one(node, rank, model_dir, &model_id, &per_rank[rank as usize],
                                   &topology, cfg, draft.as_ref())?;
        streams.push(stream);
    }
    anyhow::ensure!(streams.len() == need,
        "expected {need} retained serving control stream(s), got {}", streams.len());
    eprintln!("[head] {} node(s) synced; all control streams RETAINED for the serving session", streams.len());
    Ok((nodes, streams))
}

// ---------------------------------------------------------------------------------------------------
// Blob cache management (ops CLI: --cached-models-list / --cached-models-remove /
// --cached-models-remove-all — MODEL-centric; the old blob-centric names are deprecated aliases)
// ---------------------------------------------------------------------------------------------------

/// Walk every symlink under `cache/models/<model_id>/`, calling `f(model_id, link_path, target)`.
fn walk_model_links(mut f: impl FnMut(&str, &Path, &Path)) {
    let mdir = cache_root().join("models");
    let Ok(models) = std::fs::read_dir(&mdir) else { return };
    for e in models.flatten() {
        let mid = e.file_name().to_string_lossy().to_string();
        let mut stack = vec![e.path()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for ent in rd.flatten() {
                let p = ent.path();
                let Ok(ft) = ent.file_type() else { continue };
                if ft.is_dir() { stack.push(p); continue; }
                if ft.is_symlink() {
                    if let Ok(t) = std::fs::read_link(&p) { f(&mid, &p, &t); }
                }
            }
        }
    }
}

fn fmt_gib(b: u64) -> String { format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64) }

/// The blobs (by hash) referenced by the assembled model dir `model_id` (empty if absent).
fn model_blob_hashes(model_id: &str) -> Vec<String> {
    let mut v = Vec::new();
    walk_model_links(|mid, _p, t| {
        if mid == model_id {
            if let Some(h) = t.file_name() { v.push(h.to_string_lossy().to_string()); }
        }
    });
    v.sort(); v.dedup();
    v
}

/// Blob file size in `blobs/<hash>` (0 if missing).
fn blob_size(hash: &str) -> u64 {
    std::fs::metadata(blob_path(hash)).map(|m| m.len()).unwrap_or(0)
}

/// `--cached-models-list`: ONE line per assembled MODEL — name, total size (sum of its blob
/// files), blob count — plus a cache summary and any orphan-blob / interrupted-fetch tail.
pub fn list_cached_models() -> Result<()> {
    let mdir = cache_root().join("models");
    let mut names: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&mdir) {
        for e in rd.flatten() {
            let mid = e.file_name().to_string_lossy().to_string();
            if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) { names.push(mid); }
        }
    }
    names.sort();
    let mut rows: Vec<(String, u64, usize)> = Vec::new();
    let mut all_blobs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mid in &names {
        let hashes = model_blob_hashes(mid);
        let total: u64 = hashes.iter().map(|h| blob_size(h)).sum();
        all_blobs.extend(hashes.iter().cloned());
        rows.push((mid.clone(), total, hashes.len()));
    }
    let blob_dir = cache_root().join("blobs");
    let mut blob_total = 0u64;
    let mut n_blobs = 0usize;
    let mut tmp: Vec<(String, u64)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&blob_dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            if name.starts_with("tmp.") { tmp.push((name, size)); }
            else { blob_total += size; n_blobs += 1; }
        }
    }
    let orphans: Vec<String> = std::fs::read_dir(&blob_dir).into_iter().flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| !n.starts_with("tmp.") && !all_blobs.contains(n))
        .collect();
    println!("cached models ({}):", rows.len());
    for (mid, total, n) in &rows {
        println!("  {mid}  {:>12}  {n} blob(s)", fmt_gib(*total));
    }
    println!("cache {} — {} blob(s), {} total", blob_dir.display(), n_blobs, fmt_gib(blob_total));
    if !orphans.is_empty() {
        let osz: u64 = orphans.iter().map(|h| blob_size(h)).sum();
        println!("-- {} orphan blob(s), {} (referenced by no model; --cached-models-remove-all reclaims)", orphans.len(), fmt_gib(osz));
    }
    if !tmp.is_empty() {
        let t: u64 = tmp.iter().map(|x| x.1).sum();
        println!("-- {} interrupted-fetch partial(s), {} reclaimable (tmp.*)", tmp.len(), fmt_gib(t));
    }
    Ok(())
}

/// Match a model id by exact name or unique prefix (a 4-char floor, like the old blob op).
fn resolve_model(id: &str) -> Result<String> {
    let mdir = cache_root().join("models");
    let mut matches: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&mdir) {
        for e in rd.flatten() {
            let mid = e.file_name().to_string_lossy().to_string();
            if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                && (mid == id || mid.starts_with(id)) { matches.push(mid); }
        }
    }
    match matches.len() {
        1 => Ok(matches.pop().unwrap()),
        0 => bail!("no cached model matching '{id}' in {}", mdir.display()),
        n => bail!("'{id}' matches {n} models — give a longer prefix"),
    }
}

/// `--cached-models-remove <id>`: remove ONE cached MODEL — its assembled dir plus the blobs
/// referenced by NO other model (shared blobs stay). Interrupted-fetch partials are never
/// referenced and are reclaimed here too. Other models are untouched.
pub fn remove_cached_model(id: &str) -> Result<()> {
    if id.len() < 4 { bail!("refusing to match an id shorter than 4 characters"); }
    let mid = resolve_model(id)?;
    let hashes = model_blob_hashes(&mid);
    let mdir = cache_root().join("models").join(&mid);
    std::fs::remove_dir_all(&mdir).with_context(|| format!("remove model dir {mdir:?}"))?;
    // Garbage-collect blobs no longer referenced by ANY remaining model.
    let mut refs: std::collections::HashSet<String> = std::collections::HashSet::new();
    walk_model_links(|_m, _p, t| {
        if let Some(h) = t.file_name() { refs.insert(h.to_string_lossy().to_string()); }
    });
    let blob_dir = cache_root().join("blobs");
    let mut removed = 0u64;
    let mut n_removed = 0usize;
    if let Ok(rd) = std::fs::read_dir(&blob_dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let orphan = if name.starts_with("tmp.") { true }
                else { !refs.contains(&name) };
            if orphan {
                let sz = e.metadata().map(|m| m.len()).unwrap_or(0);
                let _ = std::fs::remove_file(e.path());
                removed += sz; n_removed += 1;
            }
        }
    }
    let kept = hashes.iter().filter(|h| blob_path(h).exists()).count();
    println!("removed model {mid} — {n_removed} unreferenced blob(s) reclaimed, {}",
             fmt_gib(removed));
    println!("  {kept} of its blob(s) still referenced by other models were kept");
    Ok(())
}

/// `--cached-models-remove-all`: clear the whole cache (blobs incl. tmp.* partials + assembled
/// model dirs). The next head run re-syncs from scratch.
pub fn remove_all_cached_models() -> Result<()> {
    let dir = cache_root().join("blobs");
    let mut total = 0u64;
    let mut n = 0usize;
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            total += e.metadata().map(|m| m.len()).unwrap_or(0);
            let _ = std::fs::remove_file(e.path());
            n += 1;
        }
    }
    let mdir = cache_root().join("models");
    if mdir.exists() { std::fs::remove_dir_all(&mdir).context("remove assembled model dirs")?; }
    println!("cleared {n} blob(s), {} — assembled model dirs removed; next head run re-syncs from scratch",
             fmt_gib(total));
    Ok(())
}

// -- deprecated aliases (the old blob-centric names; warn once, route to the model-centric ops) --

/// DEPRECATED alias for `--cached-models-list` (was: one line per BLOB). Warns once.
pub fn list_model_blobs() -> Result<()> {
    eprintln!("warning: --list-model-blobs is deprecated — use --cached-models-list (lists MODELS, not blobs)");
    list_cached_models()
}

/// DEPRECATED alias for `--cached-models-remove <model-id>` (was: remove one BLOB by hash).
/// The argument is now a MODEL name/prefix.
pub fn remove_model_blob(id: &str) -> Result<()> {
    eprintln!("warning: --remove-model-blob is deprecated — use --cached-models-remove <model> (removes a MODEL, not a blob)");
    remove_cached_model(id)
}

/// DEPRECATED alias for `--cached-models-remove-all`. Warns once.
pub fn clear_model_blobs() -> Result<()> {
    eprintln!("warning: --clear-model-blobs is deprecated — use --cached-models-remove-all");
    remove_all_cached_models()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(hostname: &str, ip: &str, port: u16) -> NodeInfo {
        NodeInfo {
            hostname: hostname.to_string(),
            addr: SocketAddr::new(ip.parse().unwrap(), port),
        }
    }

    // Synthetic RoCE rails: rail 1 = 192.168.177.0/24, rail 2 = 192.168.178.0/24.
    fn rails() -> Vec<Roce> {
        vec![
            Roce {
                ib: "rocep1s0f1".into(),
                netdev: "r1".into(),
                ip: "192.168.177.1".parse().unwrap(),
                bcast: "192.168.177.255".parse().unwrap(),
                mask: 0xffffff00,
                rail: 1,
            },
            Roce {
                ib: "roceP2p1s0f1".into(),
                netdev: "r2".into(),
                ip: "192.168.178.1".parse().unwrap(),
                bcast: "192.168.178.255".parse().unwrap(),
                mask: 0xffffff00,
                rail: 2,
            },
        ]
    }

    #[test]
    fn deterministic_rank_order_prefers_rail_then_ip() {
        let ifaces = rails();
        // rail-1 .13, rail-1 .12, rail-2 .99, non-RoCE .1 -> expected order.
        let mut nodes = vec![
            node("a", "10.0.0.1", 29500),
            node("b", "192.168.178.99", 29500),
            node("c", "192.168.177.13", 29500),
            node("d", "192.168.177.12", 29500),
        ];
        assign_ranks(&mut nodes, &ifaces);
        let order: Vec<String> = nodes.iter().map(|n| n.hostname.clone()).collect();
        assert_eq!(order, vec!["d", "c", "b", "a"], "rail desc, then ip asc");

        // Reassign with the same set in a different input order -> identical result.
        let mut shuffled = vec![
            node("d", "192.168.177.12", 29500),
            node("b", "192.168.178.99", 29500),
            node("a", "10.0.0.1", 29500),
            node("c", "192.168.177.13", 29500),
        ];
        assign_ranks(&mut shuffled, &ifaces);
        let order2: Vec<String> = shuffled.iter().map(|n| n.hostname.clone()).collect();
        assert_eq!(order, order2, "order must be input-order independent");
    }

    #[test]
    fn topology_is_full_and_indexed_by_rank() {
        let ifaces = rails();
        let mut nodes = vec![
            node("b", "192.168.178.99", 29500),
            node("c", "192.168.177.13", 29500),
            node("d", "192.168.177.12", 29500),
        ];
        assign_ranks(&mut nodes, &ifaces);
        let topo = build_topology(&nodes, &ifaces, 4).unwrap();
        assert_eq!(topo.len(), 4);
        assert_eq!(topo[0], "192.168.177.1"); // head = rail 1
        assert_eq!(topo[1], "192.168.177.12"); // rail 1, lowest ip
        assert_eq!(topo[2], "192.168.177.13"); // rail 1, next ip
        assert_eq!(topo[3], "192.168.178.99"); // rail 2 last
    }

    #[test]
    fn manifest_filtering_per_rank_and_world2_identity() {
        // Logical paths exactly as model_files() would emit relative to the model dir.
        let logicals = vec![
            "config.json",
            "inference/config.json",
            "rank1/weights.safetensors",
            "rank1/dspark_stage0.safetensors",
            "rank2/weights.safetensors",
            "rank3/weights.safetensors",
            "rank0/weights.safetensors",
        ];

        // world==2 with rank1/ present: rank1 manifest == the pre-P4 set (rank1/* + 2 root files).
        let got = |rank: usize, sharded: bool| -> Vec<&str> {
            logicals.iter().copied().filter(|rel| include_for_rank(rel, rank, sharded)).collect()
        };
        assert_eq!(got(1, true), vec![
            "config.json",
            "inference/config.json",
            "rank1/weights.safetensors",
            "rank1/dspark_stage0.safetensors",
        ]);

        // world==4 with rank1/..rank3/ present: each node gets only its own shard.
        assert_eq!(got(2, true), vec![
            "config.json",
            "inference/config.json",
            "rank2/weights.safetensors",
        ]);
        assert_eq!(got(3, true), vec![
            "config.json",
            "inference/config.json",
            "rank3/weights.safetensors",
        ]);

        // rank0's own shard must never leak into any node manifest.
        assert!(!got(1, true).contains(&"rank0/weights.safetensors"));
        assert!(!got(2, true).contains(&"rank0/weights.safetensors"));
        assert!(!got(3, true).contains(&"rank0/weights.safetensors"));

        // Replicate-if-not-divisible: a rank with no dir gets the whole model.
        let all: Vec<&str> = logicals.iter().copied().filter(|rel| include_for_rank(rel, 2, false)).collect();
        assert_eq!(all.len(), logicals.len());
    }
}
