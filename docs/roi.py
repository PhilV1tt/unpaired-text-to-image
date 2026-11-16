"""ROI scenarios for the C2 cadrage note (course formulas, section 04).

All inputs are hypotheses.
"""

import csv

MANUAL_MIN = 0.5  # min to label one image by hand (hypothesis, to be measured in E0)
# vol/an, min gagnées/cas, €/h, min revue ajoutée/cas, autres gains €, invest €,
# compute €/an, maintenance €/an, taux reprise
S: dict[str, dict[str, float]] = {
    "pessimiste": dict(
        vol=20000, gain=0.25, h=40, review=0.10, other=1500,
        invest=35000, compute=3000, maint=12000, rework=0.15,
    ),
    "central": dict(
        vol=60000, gain=0.40, h=45, review=0.05, other=4000,
        invest=25000, compute=2000, maint=8000, rework=0.05,
    ),
    "optimiste": dict(
        vol=150000, gain=0.45, h=50, review=0.03, other=8000,
        invest=20000, compute=2000, maint=6000, rework=0.02,
    ),
}  # fmt: skip
rows: list[dict[str, str | float]] = []
for name, p in S.items():
    gross = p["vol"] * p["gain"] / 60 * p["h"]
    review = p["vol"] * p["review"] / 60 * p["h"]
    rework = p["rework"] * p["vol"] * MANUAL_MIN / 60 * p["h"]
    net = gross + p["other"] - review - rework
    recurring = p["compute"] + p["maint"]
    cost1 = p["invest"] + recurring
    roi = (net - cost1) / cost1 * 100
    payback = p["invest"] / (net - recurring) * 12 if net > recurring else float("nan")
    rows.append(
        dict(
            scenario=name,
            **p,
            gain_brut=round(gross),
            cout_revue=round(review),
            cout_reprise=round(rework),
            gain_net=round(net),
            cout_annee1=round(cost1),
            roi_annee1_pct=round(roi, 1),
            retour_mois=round(payback, 1),
        )
    )
    print(name, rows[-1])
with open("02_hypotheses_roi.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# Break-even thresholds on the central scenario, written to 02_seuils_roi.csv.
c = S["central"]
net = rows[1]["gain_net"]
assert isinstance(net, int), net  # rounded net gain of the central scenario, €/an
seuils: list[dict[str, str | float]] = []
for n in range(1, 5):  # n cells share the investment and maintenance; compute is per cell
    cost1 = c["invest"] + c["maint"] + n * c["compute"]
    seuils.append(
        dict(
            levier="cellules",
            entree=n,
            seuil=round((n * net - cost1) / cost1 * 100, 1),
            unite="ROI annee 1 %",
        )
    )
gap = c["invest"] + c["compute"] + c["maint"] - net  # what one cell lacks to break even in year 1
seuils.append(dict(levier="ecart", entree="1 cellule", seuil=gap, unite="EUR manquants annee 1"))
DELTA_ACC = 0.10  # H1 threshold: accuracy points gained on held-out combinations
for picks, rare in [(100_000, 0.01), (500_000, 0.01), (1_000_000, 0.05)]:
    seuils.append(
        dict(
            levier="erreurs",
            entree=f"{picks} pieces, {rare:.0%} rares",
            seuil=round(gap / (picks * rare * DELTA_ACC), 1),
            unite="EUR par erreur evitee",
        )
    )
REFS = 120  # new references per year, same as the volume hypothesis
for days in (1, 3, 5):
    seuils.append(
        dict(
            levier="mise en service",
            entree=f"{days} j gagnes par reference",
            seuil=round(gap / (REFS * days), 1),
            unite="EUR par reference et par jour",
        )
    )
for s in seuils:
    print(s)
with open("02_seuils_roi.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(seuils[0]))
    w.writeheader()
    w.writerows(seuils)
