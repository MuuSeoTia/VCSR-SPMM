#!/usr/bin/env bash
set -euo pipefail

# ===== Config =====
ROOT_DIR="${1:-$HOME/projects/sparse-vs-dense-matmul/data}"
mkdir -p "$ROOT_DIR"
TMP="$ROOT_DIR/.tmp"; mkdir -p "$TMP"
DONE_FILE="$ROOT_DIR/.done"; touch "$DONE_FILE"
FAILED_FILE="$ROOT_DIR/.failed"; touch "$FAILED_FILE"

# Proxy (Explorer)
export http_proxy="${http_proxy:-http://10.99.0.130:3128}"
export https_proxy="${https_proxy:-http://10.99.0.130:3128}"

# How many to fetch total and how many to run in parallel
MAX_TOTAL="${MAX_TOTAL:-40}"
PARALLEL="${PARALLEL:-3}"

# Preferred groups to crawl for depth & diversity (edit freely)
GROUPS=(
  Janna GHS_psdef Williams DIMACS10 LAW SNAP Sandia Boeing vanHeukelum HB Meszaros Rothberg
)

# ===== Helpers =====
log(){ printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*" >&2; }
record_done(){ echo "$1/$2" >> "$DONE_FILE"; }
already_done(){ grep -qx "$1/$2" "$DONE_FILE" 2>/dev/null; }
already_have(){ compgen -G "$ROOT_DIR/$2/*.mtx" >/dev/null 2>&1; }

# Extract first .tar.gz link from a matrix HTML page
# Robust against layout changes: we trust *whatever* href ends with .tar.gz
html_tar_url() {
  local grp="$1" name="$2" page url
  page=$(curl -fsSL "https://sparse.tamu.edu/${grp}/${name}" || true)
  # Pull the first href="...tar.gz"
  url=$(printf '%s' "$page" | grep -oE 'href="https?://[^"]+\.tar\.gz"' | head -n1 | sed -E 's/^href="//; s/"$//')
  printf '%s' "${url:-}"
}

# Download+extract one matrix (idempotent & atomic)
fetch_one() {
  local grp="$1" name="$2"
  local base="$ROOT_DIR/$name"; mkdir -p "$base"
  if already_have "$grp" "$name"; then
    log "[SKIP] $grp/$name (already have .mtx)"; record_done "$grp" "$name"; return 0; fi
  if already_done "$grp" "$name"; then
    log "[SKIP] $grp/$name (listed in .done)"; return 0; fi

  # Find the real tarball URL from the HTML page
  local url tarpath
  url="$(html_tar_url "$grp" "$name")"
  if [[ -z "$url" ]]; then
    log "[WARN] $grp/$name: no .tar.gz link on page"; echo -e "$grp/$name\tno_tarball_link" >> "$FAILED_FILE"; return 1
  fi

  tarpath="$TMP/${grp}_${name}.tar.gz"
  log "[INFO] $grp/$name: downloading $url"
  if ! curl -fsSL --retry 6 --retry-delay 5 -o "$tarpath" "$url"; then
    log "[WARN] $grp/$name: download failed"; echo -e "$grp/$name\tdownload_failed" >> "$FAILED_FILE"; return 1
  fi

  # Extract safely
  if ! tar -xzf "$tarpath" -C "$base" 2>/dev/null; then
    log "[WARN] $grp/$name: tar failed"; echo -e "$grp/$name\ttar_failed" >> "$FAILED_FILE"; return 1
  fi

  # Locate .mtx (or .mtx.gz -> inflate)
  local mtx
  mtx=$(find "$base" -maxdepth 3 -type f -name "*.mtx" | head -n1 || true)
  if [[ -z "$mtx" ]]; then
    local gz
    gz=$(find "$base" -maxdepth 3 -type f -name "*.mtx.gz" | head -n1 || true)
    if [[ -n "$gz" ]]; then
      log "[INFO] $grp/$name: inflating $(basename "$gz")"
      gunzip -f "$gz"
      mtx="${gz%.gz}"
    fi
  fi

  if [[ -n "$mtx" ]]; then
    log "[OK]   $grp/$name → $mtx"
    record_done "$grp" "$name"
    return 0
  else
    log "[WARN] $grp/$name: no .mtx found after extract"
    echo -e "$grp/$name\tno_mtx_after_extract" >> "$FAILED_FILE"
    return 1
  fi
}

# Crawl a group page to discover matrix names (HTML scraping)
discover_group_names() {
  local grp="$1"
  # fetch group index and extract /Group/Name hrefs, then take the Name
  curl -fsSL "https://sparse.tamu.edu/${grp}" \
    | grep -oE "/${grp}/[A-Za-z0-9_.-]+" \
    | awk -F/ '{print $3}' \
    | sort -u
}

# ===== MAIN =====
log "[INFO] Destination: $ROOT_DIR"
count=0
pids=()

for grp in "${GROUPS[@]}"; do
  log "[INFO] Crawling group: $grp"
  mapfile -t names < <(discover_group_names "$grp" || true)
  if [[ ${#names[@]} -eq 0 ]]; then
    log "[WARN] $grp: no names discovered (site hiccup?)"
    continue
  fi

  for name in "${names[@]}"; do
    # Respect MAX_TOTAL across all groups
    if (( count >= MAX_TOTAL )); then break; fi
    # Avoid duplicate target directories (same 'name' under different groups)
    if already_have "$grp" "$name" || already_done "$grp" "$name"; then
      log "[SKIP] $grp/$name (already have/listed)"; continue; fi

    # Concurrency control
    while (( $(jobs -r | wc -l) >= PARALLEL )); do sleep 2; done
    fetch_one "$grp" "$name" &
    pids+=($!)
    ((count++))
  done
  if (( count >= MAX_TOTAL )); then break; fi
done

# Wait for background fetches to finish
wait

log "[INFO] Finished. Total .mtx on disk: $(find "$ROOT_DIR" -name '*.mtx' | wc -l)"
log "[INFO] .done entries: $(wc -l < "$DONE_FILE" 2>/dev/null || echo 0)   ·   .failed entries: $(wc -l < "$FAILED_FILE" 2>/dev/null || echo 0)"
