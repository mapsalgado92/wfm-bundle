import pandas as pd

from src.generators.shift_requirements import ShiftRequirements


def shift_reqs_coverage_df(requirements: ShiftRequirements) -> pd.DataFrame:

    output_df = pd.concat(
        [
            pd.Series(requirements.interval_reqs, name="reqs"),
            pd.Series(
                requirements.coverage,
                name="cover",
            ).astype("float"),
        ],
        axis=1,
    )

    return output_df.assign(diff=output_df.cover - output_df.reqs)


def shift_reqs_results_df(
    requirements: ShiftRequirements, pivoted: bool = False
) -> pd.DataFrame:
    col_names = requirements.column_names
    col_days = requirements.column_days
    col_shifts = requirements.column_shifts
    output_df = pd.concat(
        [
            pd.Series(col_names, name="shift_cols"),
            pd.Series([shift.id for shift in col_shifts], name="shift_name"),
            pd.Series(col_days, name="day_index").astype(int),
            pd.Series(requirements.weights, name="req").astype("float"),
        ],
        axis=1,
    )

    return (
        output_df
        if not pivoted
        else output_df.pivot(
            index="day_index",
            columns="shift_name",
        )
    )


def evolution_df(type: str, requirements: ShiftRequirements) -> pd.DataFrame:
    if not type in ["gd_evolution", "sr_evolution"]:
        raise ValueError("Invalid evolution type %s.", type)

    return (
        pd.DataFrame(
            [entry["values"] for entry in getattr(requirements, type)],
            columns=requirements.column_names,
        )
        .astype("float")
        .round(2)
    )
