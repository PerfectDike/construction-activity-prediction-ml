import pandas as pd
from pathlib import Path
OUTPUT_DIR = Path("/content/construction_ml_preprocessing")
raw_csv = OUTPUT_DIR / "activity_audit.csv"
df = pd.read_csv(raw_csv, dtype={"project_id": "string", "activity_id": "string", "project_type": "string", "uk_region": "string", "gmm_topic": "string", "task_type": "string"}, low_memory=False)
for c in ["original_duration", "at_completion_duration", "physical_percent_complete"]: df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ["actual_start", "actual_finish"]: df[c + "_parsed"] = pd.to_datetime(df[c], errors="coerce")
df["valid_original"] = df["original_duration"].gt(0).fillna(False)
df["valid_actual_duration"] = df["at_completion_duration"].ge(0).fillna(False)
df["has_actual_finish"] = df["actual_finish_parsed"].notna()
df["has_actual_start"] = df["actual_start_parsed"].notna()
df["marked_complete"] = df["physical_percent_complete"].eq(100).fillna(False)
df["valid_date_order"] = (df["has_actual_start"] & df["has_actual_finish"] & (df["actual_finish_parsed"] >= df["actual_start_parsed"]))
df["numeric_candidate"] = df["valid_original"] & df["valid_actual_duration"]
df["completed_candidate"] = (df["numeric_candidate"] & df["marked_complete"] & df["valid_date_order"])
df["overrun_target_unverified"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
mask = df["numeric_candidate"]
df.loc[mask, "overrun_target_unverified"] = (df.loc[mask, "at_completion_duration"] > df.loc[mask, "original_duration"]).astype("int64")
cleaned = df.loc[df["completed_candidate"]].copy()
cleaned = cleaned.drop_duplicates(["project_id", "activity_id"], keep="first")
cleaned["actual_duration_target"] = cleaned["at_completion_duration"]
cleaned["overrun_target"] = cleaned["overrun_target_unverified"].astype("int8")
keep = ["project_id", "activity_id", "project_type", "uk_region", "gmm_topic", "task_type", "original_duration", "at_completion_duration", "physical_percent_complete", "planned_start", "planned_finish", "actual_start", "actual_finish", "concurrent_activity_count", "reported_growth", "actual_duration_target", "overrun_target"]
cleaned = cleaned[keep]
cleaned_file = OUTPUT_DIR / "cleaned_completed_candidates.csv"
cleaned.to_csv(cleaned_file, index=False)
print("Cleaned file generated with", len(cleaned), "rows.")
