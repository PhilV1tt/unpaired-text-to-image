# Générer une image à partir d'un texte sans paires annotées

Projet Gen AI

## Rendu C2 (cadrage)

- Note de cadrage : [docs/02_cadrage_metier_roi_architecture.pdf](docs/02_cadrage_metier_roi_architecture.pdf) (source LaTeX : `docs/02_cadrage_metier_roi_architecture.tex`)
- Hypothèses financières : [docs/02_hypotheses_roi.csv](docs/02_hypotheses_roi.csv) et seuils de rentabilité : [docs/02_seuils_roi.csv](docs/02_seuils_roi.csv), produits par `docs/roi.py` (`cd docs && uv run roi.py`)

## Environnement

```
uv sync
uv run ruff check .
uv run pytest
```

Les jeux de données, les checkpoints ne sont pas dans le repo. L'entraînement tourne sur deux DGX Sparks. 
