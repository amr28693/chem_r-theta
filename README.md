[README.md](https://github.com/user-attachments/files/25320642/README.md)
# Angular Order Parameter Decomposition for Information Geometry on Catalytic Reaction Networks

Code and data accompanying:

> **Angular order parameter decomposition for information geometry on catalytic reaction networks**
> Anderson M. Rodriguez (2026)
> *Physical Review E* (submitted)

## What this does

A single self-contained Python script (`comparison_reaction_geometry.py`) that runs a three-way comparison of coordinate systems for information geometry on upper glycolysis (HK → PGI → PFK) in *S. cerevisiae*, using published enzyme kinetics from Teusink et al. (2000).

**Approach 1 — ODE baseline (Euclidean).** Standard mass-action kinetics. No geometry.

**Approach 2 — Cartesian Fisher–Rao (ablation).** Information geometry in concentration coordinates. The metric is catastrophically ill-conditioned (κ ~ 10¹⁴). Geodesic integration produces negative concentrations. Fails.

**Approach 3 — Polar (r, θ) decomposition (this work).** The angular order parameter maps the reaction network into a bounded 2D space. Metric conditioning drops to κ ~ 10². Geodesics are stable, physical, and analytically tractable.

## Run

```bash
python comparison_reaction_geometry.py
```

No arguments, no configuration. Runs in ~1 second.

## Outputs

| File | Description |
|------|-------------|
| `comparison_results.pdf` | 8-panel figure (vector, journal quality) |
| `comparison_results.png` | Same figure (raster, quick viewing) |
| `comparison_validation.json` | Full numerical results for reproducibility |

## Key results (from `comparison_validation.json`)

- **Conditioning improvement:** 2.1 × 10¹¹ (11.3 orders of magnitude)
- **Cartesian eigenvalue spread:** 14.1 orders of magnitude
- **Polar eigenvalue spread:** 2.8 orders of magnitude
- **Cartesian geodesic physical?** No (G6P = −54.9 mM, F6P = −69.9 mM)
- **Polar geodesic physical?** Yes
- **Geodesic curvature peak:** t = 0.139 min (geometric regime transition)
- **Geodesic curvature dip:** t ≈ 0.158 min (relaxation onto saturation manifold)

## Enzyme parameters

All kinetic parameters from Teusink et al. (2000) *Eur. J. Biochem.* **267**, 5313. V_max values converted from U/mg × 270 ≈ mM/min. See manuscript Table I for full listing and footnotes on simplifications (Ki,G6P proxy for Tps1 feedback; Hill approximation for PFK cooperativity).

## Requirements

```
numpy >= 1.17
scipy >= 1.4
matplotlib >= 3.2
```

Python 3.7+. No other dependencies.

## Related work

The polar decomposition (r, θ) is developed at the operator level in:

> A. M. Rodriguez, "An angular order parameter for drift–diffusion systems from the Fokker–Planck operator," preprint (2026).

This PRE paper applies that framework to catalytic reaction networks and demonstrates that it resolves the Fisher–Rao conditioning catastrophe.

## License

MIT

## References

- Teusink B et al. (2000) *Eur. J. Biochem.* **267**, 5313–5329
- van Eunen K et al. (2012) *PLoS Comput. Biol.* **8**, e1002483
