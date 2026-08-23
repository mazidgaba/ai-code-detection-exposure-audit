"""Pull just the files that matter out of a Kaggle notebook's output.

A finished matched-scale run leaves 4.64 GB behind, of which about 2 GB is
worth having: the training checkpoint and the corpus it was built from. The
rest is the raw DroidCollection download, which is faster to re-fetch on Kaggle
than to move across your connection twice.

This lists the output over the REST API, downloads only the wanted files, and
lays them out ready to re-upload as a Dataset on whichever account has GPU
quota left.

    python kaggle_runs/fetch_outputs.py --list  gulammazid786/matched-scale
    python kaggle_runs/fetch_outputs.py         gulammazid786/matched-scale
    python kaggle_runs/fetch_outputs.py --all   someone-else/seed-sweep

Why raw HTTP rather than the kaggle package: this repository has a directory
called kaggle/, so `import kaggle` from the project root resolves to that
directory as an implicit namespace package and shadows the real library. Going
straight to the API avoids the problem instead of tiptoeing around it.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "results")
API = "https://www.kaggle.com/api/v1"

# What a resume actually needs. Everything else is reproducible on Kaggle in
# less time than it takes to download.
WANTED = [
    "*branch_a_*_ckpt.pt",      # model + optimiser + scheduler + scaler + epoch
    "*splits.parquet",          # the training data the checkpoint belongs to
    "*eval/reports/*.json",     # small, and holds the metrics
    "*results.zip",             # if the run reached its save cell
    # The analysis modules at home re-read these. They live in artifacts/ and
    # are only copied into results.zip by the notebook's final cell, so a run
    # killed before that cell has them here and nowhere else.
    "*proba_a*.npy",
    "*labels.parquet",
]


def credentials():
    """Return (basic_auth, headers) for requests.

    Kaggle issues two kinds of credential and the newer one is not a drop-in
    replacement for the old:

      KGAT_... bearer token   KAGGLE_API_TOKEN, or ~/.kaggle/access_token.
                              Sent as an Authorization header.
      username + key          the legacy kaggle.json, sent as HTTP Basic.

    Prefer the bearer token when both are present, since that is what the
    settings page now hands out.
    """
    home = os.path.expanduser("~")
    tok = os.environ.get("KAGGLE_API_TOKEN")
    tok_path = os.path.join(home, ".kaggle", "access_token")
    if not tok and os.path.exists(tok_path):
        tok = io.open(tok_path, encoding="utf-8").read().strip()
    if tok:
        return None, {"Authorization": f"Bearer {tok}"}

    u, k = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if u and k:
        return (u, k), {}

    json_path = os.path.join(home, ".kaggle", "kaggle.json")
    if os.path.exists(json_path):
        d = json.load(io.open(json_path, encoding="utf-8"))
        return (d["username"], d["key"]), {}

    raise SystemExit(
        "No Kaggle credentials.\n\n"
        "  1. Open https://www.kaggle.com/settings -> Account -> API\n"
        "  2. Create New Token. Copy it; it is shown only once.\n"
        "  3. Save it, without a trailing newline, to:\n"
        f"       {tok_path}\n\n"
        "     PowerShell:\n"
        '       $t = "KGAT_paste_yours_here"\n'
        f'       New-Item -ItemType Directory -Force "{os.path.join(home, ".kaggle")}" | Out-Null\n'
        f'       [IO.File]::WriteAllText("{tok_path}", $t)\n\n'
        "     Or set KAGGLE_API_TOKEN in the environment instead.\n\n"
        "A legacy kaggle.json with username and key still works if you have one.\n"
        "Use the credential of the account that OWNS the notebook: a private\n"
        "notebook's output is visible only to its owner.")


def list_output(ref: str, auth, headers):
    """Every file in the notebook's output, newest version."""
    if "/" not in ref:
        raise SystemExit(f"expected <user>/<kernel-slug>, got {ref!r}")
    user, slug = ref.split("/", 1)
    files, token = [], None
    while True:
        params = {"userName": user, "kernelSlug": slug}
        if token:
            params["pageToken"] = token
        r = requests.get(f"{API}/kernels/output", params=params, auth=auth,
                         headers=headers, timeout=60)
        if r.status_code in (401, 403):
            raise SystemExit(
                f"{r.status_code} for {ref}. Either the credential is no longer\n"
                "valid, or it belongs to a different account than the notebook.\n"
                "A private notebook's output is visible only to its owner, so\n"
                "use that account's token.")
        if r.status_code == 404:
            raise SystemExit(f"404 for {ref}. Check the slug in the notebook URL.")
        r.raise_for_status()
        body = r.json()
        files.extend(body.get("files", []))
        token = body.get("nextPageToken")
        if not token:
            return files


def wanted(name: str, patterns) -> bool:
    n = name.replace("\\", "/")
    return any(fnmatch.fnmatch(n, p) for p in patterns)


def human(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def download(f, out_dir: str, auth, headers) -> str:
    """Stream to a .part file and rename, so an interrupted pull is obvious."""
    name = f["fileName"].replace("\\", "/")
    dest = os.path.join(out_dir, os.path.basename(name))
    os.makedirs(out_dir, exist_ok=True)

    size = int(f.get("size") or 0)
    if os.path.exists(dest) and size and os.path.getsize(dest) == size:
        print(f"    [have] {os.path.basename(name)}")
        return dest

    tmp = dest + ".part"
    with requests.get(f["url"], auth=auth, headers=headers, stream=True,
                      timeout=600) as r:
        r.raise_for_status()
        got = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                got += len(chunk)
                if size:
                    pct = 100.0 * got / size
                    print(f"\r    {os.path.basename(name):40s} "
                          f"{pct:5.1f}%  {human(got)}", end="", flush=True)
    print()
    os.replace(tmp, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+", metavar="USER/KERNEL-SLUG",
                    help="from the notebook URL, e.g. gulammazid786/matched-scale")
    ap.add_argument("--list", action="store_true",
                    help="show the output listing and stop")
    ap.add_argument("--all", action="store_true",
                    help="download everything, not just the resume files")
    args = ap.parse_args()

    auth, headers = credentials()
    kind = "bearer token" if headers else f"username {auth[0]}"
    print(f"authenticating with {kind}\n")

    for ref in args.refs:
        slug = ref.split("/", 1)[1]
        print("=" * 66)
        print(ref)
        print("=" * 66)
        files = list_output(ref, auth, headers)
        if not files:
            print("  no output files. The run produced nothing, which usually\n"
                  "  means it was cancelled before it wrote a checkpoint.\n")
            continue

        keep = files if args.all else [f for f in files
                                       if wanted(f["fileName"], WANTED)]
        total = sum(int(f.get("size") or 0) for f in files)
        take = sum(int(f.get("size") or 0) for f in keep)
        print(f"  {len(files)} files, {human(total)} total")
        print(f"  {len(keep)} match, {human(take)} to download\n")

        for f in sorted(files, key=lambda x: -int(x.get("size") or 0))[:25]:
            mark = "->" if f in keep else "  "
            print(f"  {mark} {human(f.get('size')):>10s}  {f['fileName']}")
        if len(files) > 25:
            print(f"     ... and {len(files) - 25} more")
        print()

        if args.list:
            continue
        if not keep:
            print("  nothing to fetch. Use --all to pull everything.\n")
            continue

        out = os.path.join(DEST, slug)
        print(f"  -> {out}")
        got = [download(f, out, auth, headers) for f in keep]

        ck = [p for p in got if p.endswith("_ckpt.pt")]
        sp = [p for p in got if p.endswith("splits.parquet")]
        print()
        if ck and sp:
            print("  Ready to re-upload. On the account with GPU quota:")
            print("    Datasets -> New Dataset -> upload these two files:")
            for p in ck + sp:
                print(f"      {p}")
            tag = os.path.basename(ck[0])[len("branch_a_"):-len("_ckpt.pt")]
            print(f"    Title it  aicd-ckpt-{tag}  and set it Private.")
            print("    Do not rename the files; the tag is parsed from the name.")
        elif not ck:
            print("  No checkpoint in this output. That run never finished an")
            print("  epoch, so it has to start over rather than resume.")
        print()


if __name__ == "__main__":
    main()
