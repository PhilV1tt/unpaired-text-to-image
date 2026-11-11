"""ROI scenarios for the C2 cadrage note (course formulas, section 04). All inputs are hypotheses."""
import csv
MANUAL_MIN = 0.5  # min to label one image by hand (hypothesis, to be measured in E0)
S = {  # vol/an, min gagnées/cas, €/h, min revue ajoutée/cas, autres gains €, invest €, compute €/an, maintenance €/an, taux reprise
 "pessimiste": dict(vol=20000, gain=0.25, h=40, review=0.10, other=1500, invest=35000, compute=3000, maint=12000, rework=0.15),
 "central":    dict(vol=60000, gain=0.40, h=45, review=0.05, other=4000, invest=25000, compute=2000, maint=8000, rework=0.05),
 "optimiste":  dict(vol=150000, gain=0.45, h=50, review=0.03, other=8000, invest=20000, compute=2000, maint=6000, rework=0.02),
}
rows = []
for name, p in S.items():
    gross = p["vol"] * p["gain"] / 60 * p["h"]
    review = p["vol"] * p["review"] / 60 * p["h"]
    rework = p["rework"] * p["vol"] * MANUAL_MIN / 60 * p["h"]
    net = gross + p["other"] - review - rework
    recurring = p["compute"] + p["maint"]
    cost1 = p["invest"] + recurring
    roi = (net - cost1) / cost1 * 100
    payback = p["invest"] / (net - recurring) * 12 if net > recurring else float("nan")
    rows.append(dict(scenario=name, **p, gain_brut=round(gross), cout_revue=round(review), cout_reprise=round(rework),
                     gain_net=round(net), cout_annee1=round(cost1), roi_annee1_pct=round(roi, 1), retour_mois=round(payback, 1)))
    print(name, rows[-1])
with open("02_hypotheses_roi.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
