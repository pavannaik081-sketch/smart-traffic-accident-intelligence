
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT / "dataset" / "raw" / "traffic_accidents.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "dataset" / "processed"
    / "traffic_accidents_dashboard.csv"
)

# Required dashboard columns (normalized names)
REQUIRED_COLUMNS = [
    "crash_date",
    "crash_time",
    "borough",
    "latitude",
    "longitude",
    "number_of_persons_injured",
    "number_of_persons_killed",
    "contributing_factor_vehicle_1",
    "vehicle_type_code_1",
]


def normalize_column(column):
    return column.strip().lower().replace(" ", "_")


def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found: {INPUT_FILE}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Starting dataset reduction...")
    print("Reading original dataset in chunks...")

    # Check actual CSV columns
    actual_columns = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()

    normalized_columns = {
        normalize_column(column): column
        for column in actual_columns
    }

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in normalized_columns
    ]

    if missing_columns:
        print("ERROR: Required columns missing:")
        print(missing_columns)
        return

    # Select original column names using normalized names
    selected_original_columns = [
        normalized_columns[column]
        for column in REQUIRED_COLUMNS
    ]

    first_chunk = True
    total_rows = 0

    try:

        for chunk in pd.read_csv(
            INPUT_FILE,
            usecols=selected_original_columns,
            chunksize=100000,
            low_memory=False
        ):

            # Normalize selected column names
            chunk.columns = [
                normalize_column(column)
                for column in chunk.columns
            ]

            # Keep columns in dashboard-required order
            chunk = chunk[REQUIRED_COLUMNS]

            chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            total_rows += len(chunk)
            first_chunk = False

            print(f"Processed rows: {total_rows:,}")

        print("\n================================")
        print("DATASET REDUCTION COMPLETED")
        print("================================")

        print(f"Total rows: {total_rows:,}")

        original_size = (
            INPUT_FILE.stat().st_size / (1024 * 1024)
        )

        reduced_size = (
            OUTPUT_FILE.stat().st_size / (1024 * 1024)
        )

        print(f"Original size: {original_size:.2f} MB")
        print(f"Reduced size: {reduced_size:.2f} MB")

        print(f"\nOutput saved at:\n{OUTPUT_FILE}")

    except Exception as error:
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
