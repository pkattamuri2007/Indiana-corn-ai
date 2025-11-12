import pandas as pd
from date_utils import parse_date_col

recs = pd.read_csv("recommendations.csv")
recs["date"] = parse_date_col(recs["date"])

# Top planting windows per location (next 30 days if your data includes future)
plant = (recs[recs["planting_recommendation"]=="Consider planting this week"]
         .sort_values(["latitude","longitude","date","planting_prob"], ascending=[True,True,True,False]))

irrig = (recs[recs["irrigation_recommendation"]=="Plan irrigation soon"]
         .sort_values(["latitude","longitude","date","irrigation_prob"], ascending=[True,True,True,False]))

# Compact summaries
plant_summary = (plant.groupby(["latitude","longitude"])
                      .agg(first_date=("date","min"),
                           last_date=("date","max"),
                           num_days=("date","nunique"),
                           max_prob=("planting_prob","max"))
                      .reset_index())

irrig_summary = (irrig.groupby(["latitude","longitude"])
                      .agg(first_alert=("date","min"),
                           last_alert=("date","max"),
                           alerts=("date","nunique"),
                           max_prob=("irrigation_prob","max"))
                      .reset_index())

plant.to_csv("planting_days_detailed.csv", index=False)
irrig.to_csv("irrigation_alerts_detailed.csv", index=False)
plant_summary.to_csv("planting_summary_by_location.csv", index=False)
irrig_summary.to_csv("irrigation_summary_by_location.csv", index=False)

print("Saved:")
print(" - planting_days_detailed.csv")
print(" - irrigation_alerts_detailed.csv")
print(" - planting_summary_by_location.csv")
print(" - irrigation_summary_by_location.csv")
