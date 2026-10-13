"""Independent validation of the option pricer. Writes docs/validation_report.md and tables under examples/output/."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# == config
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "output"
REPORT_PATH = ROOT / "docs" / "validation_report.md"
MC_PATHS = 200_000
FD_GRID = 800
HEDGE_PATHS = 2000
# ==
sys.path.insert(0, str(ROOT / "src"))

from frtb_val.validation.model import FrontOfficePricer  # noqa: E402
from frtb_val.validation.report import render_report  # noqa: E402
from frtb_val.validation.run import run_validation  # noqa: E402


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    res = run_validation(FrontOfficePricer(), mc_paths=MC_PATHS, fd_grid=FD_GRID, hedge_paths=HEDGE_PATHS)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(res), encoding="utf-8")
    res.benchmark.to_csv(OUTPUT_DIR / "validation_benchmark_european.csv", index=False)
    res.american.to_csv(OUTPUT_DIR / "validation_benchmark_american.csv", index=False)
    res.greeks.to_csv(OUTPUT_DIR / "validation_greeks.csv", index=False)
    res.pricing_outcomes.to_csv(OUTPUT_DIR / "validation_pricing_outcomes.csv", index=False)
    res.monitoring.to_csv(OUTPUT_DIR / "validation_monitoring.csv", index=False)
    res.findings.to_csv(OUTPUT_DIR / "validation_findings.csv", index=False)
    print(res.monitoring.to_string(index=False, float_format=lambda x: f"{x:,.6g}"))
    print(res.findings[["id", "severity", "area"]].to_string(index=False))
    print("Conclusion:", res.conclusion)
    print(f"Report written to {REPORT_PATH} in {time.time() - start:.0f}s")
    return {"results": res}


if __name__ == "__main__":
    main()
