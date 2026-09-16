import csv, json
rows = list(csv.DictReader(open("datasets/data/intervention_synthetic.csv")))
out = [{
    "amount": float(r["amount"]), "deadline_hours": float(r["deadline_hours"]),
    "is_recurring": bool(int(r["is_recurring"])), "time_of_day": int(r["time_of_day"]),
    "day_of_week": int(r["day_of_week"]), "label": int(r["intervention_label"]),
} for r in rows]
json.dump(out, open("datasets/intervention_train.json", "w"))
print(f"Converted {len(out)} rows")