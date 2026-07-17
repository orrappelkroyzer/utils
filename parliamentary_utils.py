import sys
from datetime import date, datetime
from pathlib import Path

local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(config_path=Path(local_python_path) / "config.json")

import json
import re

import pandas as pd


EMPTY_PACKAGE_LABEL = "∅"
ENACTED_TEXT_KEYWORDS = (
    "adopted",
    "enacted",
    "approved",
    "implemented",
    "effective",
    "carried",
    "overridden",
    "agreed for implementation",
    "deemed adopted",
)
REJECTED_TEXT_KEYWORDS = (
    "rejected",
    "failed",
    "lapsed",
    "dropped",
    "expired",
    "not_tabled",
    "dissolution",
    "interrupted",
    "struck down",
    "not completed",
    "not passed",
    "survived",
    "no identifiable merits vote",
)


def get_package_label_from_issue_indices(issue_indices):
    if not issue_indices:
        return EMPTY_PACKAGE_LABEL
    return "+".join(chr(65 + issue_idx) for issue_idx in sorted(issue_indices))


def parse_vote_date(vote_date_value):
    if vote_date_value is None or pd.isna(vote_date_value):
        return None
    text = str(vote_date_value).strip()
    if not text:
        return None
    first_segment = text.split(";")[0].strip()
    for fmt in ("%d/%m/%Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(first_segment[:30], fmt).date()
        except ValueError:
            continue
    match = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", first_segment)
    if match is None:
        return None
    day, month, year = match.groups()
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def normalize_outcome_text(vote_result, event_status):
    return f"{vote_result or ''} {event_status or ''}".strip().lower()


def is_issue_enacted(vote_result, event_status):
    outcome_text = normalize_outcome_text(vote_result, event_status)
    if not outcome_text:
        return False
    if any(keyword in outcome_text for keyword in REJECTED_TEXT_KEYWORDS):
        return False
    return any(keyword in outcome_text for keyword in ENACTED_TEXT_KEYWORDS)


def get_issue_sort_key(issue_index, vote_date_value, item_number):
    parsed_vote_date = parse_vote_date(vote_date_value)
    missing_date_rank = 1 if parsed_vote_date is None else 0
    fallback_date = date.max if parsed_vote_date is None else parsed_vote_date
    return missing_date_rank, fallback_date, item_number, issue_index


def build_enactment_path_package_labels(
    issue_vote_dates,
    issue_vote_results,
    issue_event_statuses,
    issue_item_numbers=None,
):
    issue_count = len(issue_vote_dates)
    if issue_count == 0:
        return [EMPTY_PACKAGE_LABEL]

    if issue_item_numbers is None:
        issue_item_numbers = list(range(1, issue_count + 1))

    issue_order = sorted(
        range(issue_count),
        key=lambda issue_index: get_issue_sort_key(
            issue_index,
            issue_vote_dates[issue_index],
            issue_item_numbers[issue_index],
        ),
    )

    enacted_indices = set()
    enactment_path = [EMPTY_PACKAGE_LABEL]
    for issue_index in issue_order:
        if is_issue_enacted(
            issue_vote_results[issue_index],
            issue_event_statuses[issue_index],
        ):
            enacted_indices.add(issue_index)
        enactment_path.append(get_package_label_from_issue_indices(enacted_indices))
    return enactment_path


def build_enactment_vote_package_label_pairs(
    issue_vote_dates,
    issue_vote_results,
    issue_event_statuses,
    issue_item_numbers=None,
    issue_vote_observation_types=None,
):
    issue_count = len(issue_vote_dates)
    if issue_item_numbers is None:
        issue_item_numbers = list(range(1, issue_count + 1))
    issue_order = sorted(
        range(issue_count),
        key=lambda issue_index: get_issue_sort_key(
            issue_index,
            issue_vote_dates[issue_index],
            issue_item_numbers[issue_index],
        ),
    )
    enacted_indices = set()
    vote_package_label_pairs = []
    for issue_index in issue_order:
        if (
            issue_vote_observation_types is not None
            and str(issue_vote_observation_types[issue_index]).lower().startswith(
                "counterfactual"
            )
        ):
            continue
        source_label = get_package_label_from_issue_indices(enacted_indices)
        proposed_indices = enacted_indices | {issue_index}
        target_label = get_package_label_from_issue_indices(proposed_indices)
        vote_package_label_pairs.append([source_label, target_label])
        if is_issue_enacted(
            issue_vote_results[issue_index],
            issue_event_statuses[issue_index],
        ):
            enacted_indices.add(issue_index)
    return vote_package_label_pairs


def build_enactment_path_from_issue_rows(country_issues):
    issue_vote_dates = country_issues["vote_date"].tolist()
    issue_vote_results = country_issues["vote_result"].tolist()
    issue_event_statuses = country_issues["event_status"].tolist()
    issue_item_numbers = country_issues["item_number_within_country"].tolist()
    return build_enactment_path_package_labels(
        issue_vote_dates,
        issue_vote_results,
        issue_event_statuses,
        issue_item_numbers=issue_item_numbers,
    )


def build_enactment_vote_pairs_from_issue_rows(country_issues):
    issue_vote_observation_types = None
    if "vote_observation_type" in country_issues.columns:
        issue_vote_observation_types = country_issues[
            "vote_observation_type"
        ].tolist()
    return build_enactment_vote_package_label_pairs(
        country_issues["vote_date"].tolist(),
        country_issues["vote_result"].tolist(),
        country_issues["event_status"].tolist(),
        issue_item_numbers=country_issues["item_number_within_country"].tolist(),
        issue_vote_observation_types=issue_vote_observation_types,
    )


def get_enactment_path_node_indices(package_labels, enactment_path_package_labels):
    label_to_node_index = {
        package_label: node_index
        for node_index, package_label in enumerate(package_labels)
    }
    node_indices = []
    for package_label in enactment_path_package_labels:
        if package_label not in label_to_node_index:
            raise ValueError(
                f"Enactment path label {package_label!r} is missing from "
                f"package labels: {package_labels}"
            )
        node_indices.append(label_to_node_index[package_label])
    return node_indices


def get_enactment_path_edges(enactment_path_node_indices):
    enactment_edges = []
    for source_index in range(len(enactment_path_node_indices) - 1):
        source_node = enactment_path_node_indices[source_index]
        target_node = enactment_path_node_indices[source_index + 1]
        if source_node == target_node:
            continue
        enactment_edges.append((source_node, target_node))
    return enactment_edges


def get_enactment_path_from_scenario_row(row):
    if "enactment_path_package_labels_json" not in row:
        return None
    if pd.isna(row["enactment_path_package_labels_json"]):
        return None
    if "package_labels_json" not in row or pd.isna(row["package_labels_json"]):
        return None
    package_labels = json.loads(row["package_labels_json"])
    enactment_path_package_labels = json.loads(
        row["enactment_path_package_labels_json"],
    )
    enactment_path_node_indices = get_enactment_path_node_indices(
        package_labels,
        enactment_path_package_labels,
    )
    vote_edges = get_enactment_path_edges(enactment_path_node_indices)
    if (
        "enactment_vote_package_label_pairs_json" in row
        and not pd.isna(row["enactment_vote_package_label_pairs_json"])
    ):
        vote_package_label_pairs = json.loads(
            row["enactment_vote_package_label_pairs_json"],
        )
        vote_edges = [
            tuple(
                get_enactment_path_node_indices(package_labels, package_label_pair)
            )
            for package_label_pair in vote_package_label_pairs
        ]
    return {
        "package_labels": enactment_path_package_labels,
        "node_indices": enactment_path_node_indices,
        "edges": get_enactment_path_edges(enactment_path_node_indices),
        "vote_edges": vote_edges,
    }
