"""MSCE CLI — multi-source cross-validation from the command line."""

import argparse
import json
import sys
from pathlib import Path

from .core import check, load_hubble_data


def main():
    parser = argparse.ArgumentParser(
        prog="msce",
        description="MSCE — Multi-Source Consistency Engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # msce check
    check_parser = sub.add_parser("check", help="Run a multi-condition cross-validation check")
    check_parser.add_argument("target", help="'hubble' (built-in demo) or path to .json config")
    check_parser.add_argument("--quick", action="store_true", help="Use cached results (fast)")
    check_parser.add_argument("--output", "-o", help="Save results to file")
    check_parser.add_argument("--format", choices=["text", "json", "html"], default="text")

    # msce hubble
    sub.add_parser("hubble", help="Alias for: msce check hubble --quick")

    # msce version
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "version":
        from . import __version__
        print(f"msce {__version__}")
        return

    if args.command == "hubble":
        args.target = "hubble_data"
        args.quick = True
        args.format = "text"
        args.output = None
        run_hubble_quick(args)
        return

    if args.command == "check":
        if args.target in ("hubble", "hubble_data"):
            run_hubble_quick(args)
        else:
            run_check_file(args)


def run_hubble_quick(args):
    """Run the built-in Hubble tension analysis."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    data = load_hubble_data()

    console.print()
    console.print(Panel.fit(
        "[bold]Hubble Tension Cross-Validation Report[/]\n"
        "6 mainstream H₀ solutions × 8 independent verification conditions",
        border_style="red"
    ))

    # Summary table
    table = Table(title="Single-Proposal Results")
    table.add_column("Proposal", style="cyan")
    table.add_column("Passes", justify="center")
    table.add_column("Violations", justify="center", style="red")
    table.add_column("MSCE Confidence", justify="right")

    for key, info in data["proposals"].items():
        r = data.get("results", {}).get(key, {})
        table.add_row(
            info["name"],
            str(r.get("pass_count", "?")),
            str(r.get("violation_count", "?")),
            f"{r.get('msce_confidence', 0):.3f}",
        )

    console.print(table)

    # Verdict
    all_conf = [r.get("msce_confidence", 0) for r in data.get("results", {}).values()]
    max_conf = max(all_conf) if all_conf else 0
    console.print()
    if max_conf < 0.36:
        console.print(f"[bold red]VERDICT: All 6 proposals fail (max confidence = {max_conf:.3f}).[/]")
        console.print("[red]No single-factor solution satisfies all 8 verification conditions simultaneously.[/]")
        console.print("[yellow]Combination search: even 2-factor combinations perform worse than singles.[/]")
        console.print()
        console.print("[dim]Run the full notebook: notebooks/hubble_tension.ipynb[/]")
    else:
        console.print(f"[green]Best proposal confidence: {max_conf:.3f}[/]")

    if args.output:
        out = {"msce_version": "0.1.0", "hubble_results": data}
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        console.print(f"\n[dim]Saved to {args.output}[/]")


def run_check_file(args):
    print(f"Loading verification config from: {args.target}")
    print("(Custom file verification coming in v0.2.0)")
    print("For now, try: msce check hubble --quick")


if __name__ == "__main__":
    main()
