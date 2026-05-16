#!/usr/bin/env python3
"""
Contract Discovery — downloads DeFi fork contracts for batch scanning.

Two modes:
  1. Config-driven: KNOWN_COMPOUND_FORKS / KNOWN_AAVE_FORKS with GitHub URL or explorer address
  2. Explorer API: download verified source from chain explorers (no key needed for some)

Output: populates contracts/discovered/<name>/ directories with .sol files.
"""

import os, sys, json, time, re
import urllib.request
import urllib.error


# ═══════════════════════════════════════════════════════════════
# Known Compound v2 forks — (name, chain, comptroller_address, github_url)
# URLs verified as of 2025-05. Explorer download tried first, GitHub as fallback.
# ═══════════════════════════════════════════════════════════════

KNOWN_COMPOUND_FORKS = [
    # Ethereum mainnet
    {
        'name': 'IronBank',
        'chain': 'ethereum',
        'comptroller': '0xAB1c342C7bf5Ec5F02ADEA1c2270670bCa144CbB',
        'github': 'https://raw.githubusercontent.com/ibdotxyz/ib-v2/main/src/protocol/pool/Comptroller.sol',
    },
    {
        'name': 'RariFuse',
        'chain': 'ethereum',
        'comptroller': '0xe16db319d9da7ce40b666dd2e365a4b8b3c18217',
        'github': 'https://raw.githubusercontent.com/Rari-Capital/fuse-v1/master/contracts/Comptroller.sol',
    },
    # BSC — Venus (isolated pools repo, not the diamond proxy core)
    {
        'name': 'Venus',
        'chain': 'bsc',
        'comptroller': '0xfD36E2c2a6789Db23113685031d7F16329158384',
        'github': 'https://raw.githubusercontent.com/VenusProtocol/isolated-pools/main/contracts/Comptroller.sol',
    },
    # Avalanche — Benqi (verified on Snowtrace, no key needed)
    {
        'name': 'Benqi',
        'chain': 'avalanche',
        'comptroller': '0xb17e06929dc3b39ba3f71882b0f5d16a183bbb2f',
        'github': 'https://raw.githubusercontent.com/Benqi-fi/BENQI-Smart-Contracts/main/contracts/Comptroller.sol',
    },
    # Arbitrum — Lodestar
    {
        'name': 'Lodestar',
        'chain': 'arbitrum',
        'comptroller': '0x21490f9daad9213cd81f875e2a95e080666a3b38',
        'github': 'https://raw.githubusercontent.com/LodestarFinance/lodestar-protocol/main/contracts/Comptroller.sol',
    },
    # Polygon — 0vix (verified on Polygonscan, requires API key)
    {
        'name': '0vix',
        'chain': 'polygon',
        'comptroller': '0xf29d0ae1a29c453df338c5eee4f010cfe08bb3ff',
        'github': 'https://raw.githubusercontent.com/0vix/0vix-protocol/main/contracts/Comptroller.sol',
    },
    # Fantom — Scream (repo likely archived, try explorer)
    {
        'name': 'Scream',
        'chain': 'fantom',
        'comptroller': '0x260e596dAbE3AFc463e75B6cC05d8dA46aF1c328',
        'github': 'https://raw.githubusercontent.com/Scream-Finance/scream-v1/master/contracts/Comptroller.sol',
    },
    # Moonbeam — Moonwell
    {
        'name': 'Moonwell',
        'chain': 'moonbeam',
        'comptroller': '0x0f3905592580c3B8Cb1b7bd0F446A5331a736846',
        'github': 'https://raw.githubusercontent.com/moonwell-fi/moonwell-contracts-v2/main/src/core/Comptroller.sol',
    },
    # Harmony — Tranquil Finance
    {
        'name': 'Tranquil',
        'chain': 'harmony',
        'comptroller': '0x4Cf215fA1077Cb3f2eBeeB7A502c920c70773383',
        'github': 'https://raw.githubusercontent.com/Tranquil-Finance/tranquil-contracts/main/contracts/Comptroller.sol',
    },
    # Cronos — Tectonic
    {
        'name': 'Tectonic',
        'chain': 'cronos',
        'comptroller': '0xb3831584acb95ED9cCb0C11f677B5AD01DeaeEc0',
        'github': 'https://raw.githubusercontent.com/Tectonic-Finance/tectonic-core/main/contracts/Comptroller.sol',
    },
    # ── Additional forks ──
    # Fantom — Geist (Aave fork on Fantom)
    {
        'name': 'Geist',
        'chain': 'fantom',
        'comptroller': '0x9FAD24f572845cE1dE94a858231bD342a9eAA306',
    },
    # Aurora — Aurigami (Compound fork on NEAR/Aurora)
    {
        'name': 'Aurigami',
        'chain': 'aurora',
        'comptroller': '0x817af6cfAF5B4c068E732E348593fd473499760b',
    },
    # Hundred Finance — addresses return empty source (contracts unverified or moved)
    # ── High-priority: exploited or unaudited forks ──
    # Optimism — Sonne Finance (EXPLOITED May 2024, $20M donation attack)
    {
        'name': 'Sonne',
        'chain': 'optimism',
        'comptroller': '0x60cf091cd3f50420d50fd7f707414d0df4751c58',
    },
    # Base — Moonwell (heavily used, Cyfrin audited but still worth scanning)
    {
        'name': 'Moonwell_Base',
        'chain': 'base',
        'comptroller': '0xfBb21d0380beE3312B33c4353c8936a0F13EF26C',
    },
    # Base — Seamless Protocol is an Aave v3 fork (not Compound), skip
    # Sonic — fMoney, Compound v2 fork (ComptrollerV2)
    {
        'name': 'fMoney',
        'chain': 'sonic',
        'comptroller': '0xca1d4759159ff2577c3e7e5a5fef3069c6146b1c',
    },
    # Sonic — Enclabs, Compound v2 fork
    {
        'name': 'Enclabs',
        'chain': 'sonic',
        'comptroller': '0x5c12739ce2b0244a6e0305d58e57758c4c03ab64',
    },
    # Linea — Mendi Finance, Compound v2 fork
    {
        'name': 'Mendi',
        'chain': 'linea',
        'comptroller': '0x1b4d3b0421ddc1eb216d230bc01527422fb93103',
    },
    # Blast — ComptrollerReward, Compound v2 fork with V9 storage
    {
        'name': 'BlastReward',
        'chain': 'blast',
        'comptroller': '0xebc9005f3c272b441a5166402bc9ef58208f636a',
    },
    # Celo — Moola Protocol, Compound v2 fork
    {
        'name': 'Moola',
        'chain': 'celo',
        'comptroller': '0x9BD4Fd10b531ae07437676dfE3FA6f505032CB64',
    },
    # Avalanche — Joe Lend (Banker Joe), Trader Joe's lending protocol
    {
        'name': 'JoeLend',
        'chain': 'avalanche',
        'comptroller': '0x8B33E813e6757F5c1A5E662333463C2aB23d99b7',
    },
    # Ethereum — Onyx Protocol (EXPLOITED Nov 2023 & Sep 2024, rounding bug)
    {
        'name': 'Onyx',
        'chain': 'ethereum',
        'comptroller': '0x3047d790879714930e83b7a7d8e76c2bb64d87b9',
    },
]

KNOWN_AAVE_FORKS = [
    {'name': 'AaveV3_Base', 'chain': 'base',
     'pool': '0xA238Dd80C259a72e81d7e4664a9801593F98d1c5'},
    {'name': 'AaveV3_Arbitrum', 'chain': 'arbitrum',
     'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD'},
    {'name': 'AaveV3_Avalanche', 'chain': 'avalanche',
     'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD'},
    {'name': 'AaveV3_Optimism', 'chain': 'optimism',
     'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD'},
    {'name': 'AaveV3_Polygon', 'chain': 'polygon',
     'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD'},
]


# ═══════════════════════════════════════════════════════════════
# Download helpers
# ═══════════════════════════════════════════════════════════════

OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contracts', 'discovered')

# Chain IDs for Etherscan V2 API (one key works across all)
# https://api.etherscan.io/v2/api?chainid={id}&module=contract&action=getsourcecode&address={addr}&apikey={key}
CHAIN_IDS = {
    'ethereum': 1,
    'bsc': 56,
    'polygon': 137,
    'avalanche': 43114,
    'arbitrum': 42161,
    'sonic': 146,
    'linea': 59144,
    'blast': 81457,
    'celo': 42220,
    'optimism': 10,
    'base': 8453,
    'fantom': 250,
}

# Fallback V1 APIs for chains not in Etherscan V2 family
EXPLORER_APIS_V1 = {
    'avalanche': 'https://api.snowtrace.io/api',
    'cronos': 'https://api.cronoscan.com/api',
    'moonbeam': 'https://api-moonbeam.moonscan.io/api',
    'aurora': 'https://api.aurorascan.dev/api',
    'harmony': 'https://api.harmony.one/api',
}


def download_url(url: str, output_path: str, retries: int = 3) -> bool:
    """Download a file from URL with retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CIS-Scanner/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
                if len(content) > 100:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
    return False


def download_etherscan_source(chain: str, address: str, output_dir: str,
                              api_key: str = '') -> int:
    """Download verified source from chain explorer. Returns number of files saved.

    Uses Etherscan V2 API (one key works across all chains). Falls back to
    chain-specific V1 APIs for chains not in the Etherscan family.
    """
    urls_to_try = []

    # Primary: Etherscan V2 API
    chain_id = CHAIN_IDS.get(chain)
    if chain_id and api_key:
        urls_to_try.append(
            f'https://api.etherscan.io/v2/api?chainid={chain_id}'
            f'&module=contract&action=getsourcecode&address={address}&apikey={api_key}'
        )

    # Fallback: chain-specific V1 API (works without key for some like Snowtrace)
    v1_url = EXPLORER_APIS_V1.get(chain)
    if v1_url:
        params = {
            'module': 'contract',
            'action': 'getsourcecode',
            'address': address,
            'apikey': api_key,
        }
        query = '&'.join(f'{k}={v}' for k, v in params.items())
        urls_to_try.append(f'{v1_url}?{query}')

    if not urls_to_try:
        return 0

    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CIS-Scanner/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception:
            continue

        if data.get('status') != '1' or not data.get('result'):
            continue

        result = data['result'][0]
        source = result.get('SourceCode', '')
        contract_name = result.get('ContractName', 'Unknown')

        if not source:
            continue

        os.makedirs(output_dir, exist_ok=True)
        count = 0

        # Handle multi-file contracts — two formats:
        # V2 API: {"File.sol": {"content": "..."}, ...}
        # V1 API: {{"sources": {"File.sol": {"content": "..."}}}}
        src_stripped = source.strip()
        multi_file = None

        # Try V2 format (plain JSON object with .sol keys)
        if src_stripped.startswith('{') and '.sol' in src_stripped[:200]:
            try:
                multi_file = json.loads(src_stripped)
            except json.JSONDecodeError:
                pass

        # Try V1 format (double-brace wrapped)
        if multi_file is None and src_stripped.startswith('{{'):
            try:
                multi_file = json.loads(source[1:-1]).get('sources')
            except (json.JSONDecodeError, KeyError):
                pass

        if multi_file:
            for filepath, content in multi_file.items():
                if isinstance(content, dict):
                    code = content.get('content', '')
                else:
                    code = str(content)
                fname = os.path.basename(filepath)
                out_path = os.path.join(output_dir, fname)
                with open(out_path, 'w') as f:
                    f.write(code)
                count += 1
            return count

        # Single file
        out_path = os.path.join(output_dir, f'{contract_name}.sol')
        with open(out_path, 'w') as f:
            f.write(source)
        return 1

    return 0


# ═══════════════════════════════════════════════════════════════
# Main download orchestration
# ═══════════════════════════════════════════════════════════════

def download_known_forks(forks: list[dict], label: str, api_key: str = '') -> int:
    """Download fork contracts. Returns total files saved."""
    total = 0

    for fork in forks:
        name = fork['name']
        chain = fork['chain']
        out_dir = os.path.join(OUTPUT_BASE, f"{name}_{chain}")

        # Check if already downloaded
        existing = []
        if os.path.isdir(out_dir):
            existing = [f for f in os.listdir(out_dir) if f.endswith('.sol')]
        if existing:
            print(f"  {name} ({chain}): already have {len(existing)} files, skip")
            continue

        # Determine the contract address to download
        addr = fork.get('comptroller') or fork.get('pool')
        gh_url = fork.get('github')

        # Strategy: try Explorer API first (more reliable), then GitHub raw URL
        saved = 0
        if addr:
            saved = download_etherscan_source(chain, addr, out_dir, api_key)
            if saved:
                print(f"  {name} ({chain}): OK via explorer ({saved} files)")
                total += saved
                continue

        if gh_url:
            fname = os.path.basename(gh_url)
            out_path = os.path.join(out_dir, fname)
            if download_url(gh_url, out_path):
                print(f"  {name} ({chain}): OK via GitHub ({fname})")
                total += 1
                continue

        print(f"  {name} ({chain}): FAILED (no source available)")
        # Create empty marker so we don't retry every run
        os.makedirs(out_dir, exist_ok=True)

    return total


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    api_key = os.environ.get('ETHERSCAN_API_KEY', '')

    print("=== Contract Discovery ===\n")

    print("Compound v2 Forks:")
    c_count = download_known_forks(KNOWN_COMPOUND_FORKS, "Compound", api_key)
    print(f"  Downloaded: {c_count} files from {len(KNOWN_COMPOUND_FORKS)} known forks\n")

    print("Aave v3 Forks:")
    a_count = download_known_forks(KNOWN_AAVE_FORKS, "Aave", api_key)
    print(f"  Downloaded: {a_count} files from {len(KNOWN_AAVE_FORKS)} known forks\n")

    # Summary
    print(f"=== Summary ===")
    if os.path.isdir(OUTPUT_BASE):
        protocols = []
        for d in sorted(os.listdir(OUTPUT_BASE)):
            full = os.path.join(OUTPUT_BASE, d)
            if os.path.isdir(full):
                files = [f for f in os.listdir(full) if f.endswith('.sol')]
                if files:
                    sizes = sum(os.path.getsize(os.path.join(full, f)) for f in files)
                    print(f"  {d}: {len(files)} files, {sizes:,} bytes")
                    protocols.append(d)
        print(f"\n  {len(protocols)} protocols with contracts downloaded")
    else:
        print("  No discovered contracts yet")

    print(f"\nTotal: {c_count + a_count} files downloaded")
    print(f"Output: {OUTPUT_BASE}")
