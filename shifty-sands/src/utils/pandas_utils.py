import pandas as pd

from src.shift_requirements import ShiftRequirements


def shift_reqs_coverage_df(instance: ShiftRequirements) -> pd.DataFrame:
    last_gen = instance.last_generated

    output_df = pd.concat(
        [
            pd.Series(last_gen["interval_reqs"], name="reqs"),
            pd.Series(
                ShiftRequirements.total_coverage(
                    last_gen["weights"], last_gen["shift_matrix"]
                ),
                name="cover",
            ),
        ],
        axis=1,
    ).astype("float")

    return output_df.assign(diff=output_df.cover - output_df.reqs)


def shift_reqs_results_df(
    instance: ShiftRequirements, pivoted: bool = False
) -> pd.DataFrame:
    last_gen = instance.last_generated
    col_names = instance.shift_matrix_cols
    output_df = pd.concat(
        [
            pd.Series(col_names, name="shift_cols"),
            pd.Series([c.split("-")[0] for c in col_names], name="shift_name"),
            pd.Series([c.split("-")[1] for c in col_names], name="day_index").astype(
                int
            ),
            pd.Series(last_gen["weights"], name="req").astype("float"),
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


def evolution_df(type: str, instance: ShiftRequirements) -> pd.DataFrame:
    if not type in ["gd_evolution", "sr_evolution"]:
        raise ValueError("Invalid evolution type %s.", type)
    last_gen = instance.last_generated

    return pd.DataFrame(
        [entry["values"] for entry in last_gen[type]],
        columns=instance.shift_matrix_cols,
    ).astype("float")
