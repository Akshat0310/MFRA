from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import median


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "raw" / "mutual_funds_data2.csv"
MASTER_DATASET_PATH = ROOT_DIR / "data" / "raw" / "mutual_fund_data1.csv"
ARCHIVE_NAV_DIR = ROOT_DIR / "data" / "raw" / "archive" / "DailyNAV"
PROCESSED_DATASET_PATH = ROOT_DIR / "data" / "processed" / "fund_schemes_canonical.csv"

GENERIC_AMC_WORDS = (
    "asset management company",
    "asset management",
    "funds management",
    "mutual fund",
    "amc",
    "company",
    "limited",
    "private",
    "india",
    "co",
    "ltd",
    "pvt",
    "mgmt",
)

AMC_CANONICAL_MAP = {
    "aditya birla sun life mutual fund": "aditya birla sun life",
    "aditya birla sun life amc limited": "aditya birla sun life",
    "axis mutual fund": "axis",
    "axis asset management co ltd": "axis",
    "bandhan mutual fund": "bandhan",
    "bandhan amc limited": "bandhan",
    "bank of india mutual fund": "bank of india",
    "bank of india investment managers private limited": "bank of india",
    "baroda bnp paribas mutual fund": "baroda bnp paribas",
    "baroda bnp paribas asset management india pvt ltd": "baroda bnp paribas",
    "canara robeco mutual fund": "canara robeco",
    "canara robeco asset management company limited": "canara robeco",
    "dsp mutual fund": "dsp",
    "dsp asset managers private limited": "dsp",
    "edelweiss mutual fund": "edelweiss",
    "edelweiss asset management limited": "edelweiss",
    "franklin templeton mutual fund": "franklin templeton",
    "franklin templeton asset management india private limited": "franklin templeton",
    "hdfc mutual fund": "hdfc",
    "hdfc asset management company limited": "hdfc",
    "hsbc mutual fund": "hsbc",
    "hsbc asset management india private ltd": "hsbc",
    "icici prudential mutual fund": "icici prudential",
    "icici prudential asset management company limited": "icici prudential",
    "invesco mutual fund": "invesco",
    "invesco asset management india private limited": "invesco",
    "iti mutual fund": "iti",
    "iti asset management limited": "iti",
    "kotak mahindra mutual fund": "kotak mahindra",
    "kotak mahindra asset management company limited": "kotak mahindra",
    "lic mf mutual fund": "lic",
    "lic mutual fund": "lic",
    "lic mutual fund asset management limited": "lic",
    "mahindra manulife mutual fund": "mahindra manulife",
    "mahindra manulife investment management private limited": "mahindra manulife",
    "mirae asset mutual fund": "mirae asset",
    "mirae asset investment managers india private limited": "mirae asset",
    "motilal oswal mutual fund": "motilal oswal",
    "motilal oswal asset management company limited": "motilal oswal",
    "nippon india mutual fund": "nippon india",
    "nippon life india asset management limited": "nippon india",
    "pgim india mutual fund": "pgim india",
    "pgim india asset management private limited": "pgim india",
    "pgim india asset management private limite": "pgim india",
    "ppfas mutual fund": "parag parikh",
    "parag parikh mutual fund": "parag parikh",
    "ppfas asset management private limited": "parag parikh",
    "quant mutual fund": "quant",
    "quant money managers limited": "quant",
    "samco mutual fund": "samco",
    "samco asset management private limited": "samco",
    "sbi mutual fund": "sbi",
    "sbi funds management limited": "sbi",
    "sundaram mutual fund": "sundaram",
    "sundaram asset management company ltd": "sundaram",
    "tata mutual fund": "tata",
    "tata asset management limited": "tata",
    "taurus mutual fund": "taurus",
    "taurus asset management company limited": "taurus",
    "trust mutual fund": "trust",
    "trust asset management private limited": "trust",
    "uti mutual fund": "uti",
    "uti asset mgmt co ltd": "uti",
    "whiteoak capital mutual fund": "whiteoak capital",
    "whiteoak capital asset management limited": "whiteoak capital",
}

SCHEME_REPLACEMENTS = {
    " sl ": " sun life ",
    " fof ": " fund of fund ",
    " psu ": " public sector undertaking ",
    " govt ": " government ",
}

SCHEME_STOPWORDS = {
    "fund",
    "plan",
    "direct",
    "regular",
    "growth",
    "idcw",
    "option",
    "dividend",
    "payout",
    "reinvestment",
    "dir",
    "bonus",
    "institutional",
}


@dataclass(frozen=True)
class MasterSchemeRecord:
    scheme_code: str
    scheme_name: str
    amc: str
    scheme_type: str
    scheme_category: str
    scheme_nav_name: str
    scheme_min_amount: int | None
    nav: float | None
    latest_nav_date: str | None
    average_aum_cr: float | None
    launch_date: str | None
    closure_date: str | None
    canonical_amc_name: str
    normalized_scheme_name: str
    plan_style: str


@dataclass(frozen=True)
class ArchiveNavRecord:
    scheme_code: str
    scheme_name: str
    nav: float
    date: str
    normalized_scheme_name: str
    plan_style: str


@dataclass(frozen=True)
class FundScheme:
    scheme_name: str
    amc_name: str
    category: str
    sub_category: str
    risk_level: int
    rating: int
    min_sip: int
    min_lumpsum: int
    expense_ratio: float
    fund_size_cr: float
    fund_age_yr: float
    returns_1yr: float | None
    returns_3yr: float | None
    returns_5yr: float | None
    scheme_code: str | None = None
    nav: float | None = None
    latest_nav_date: str | None = None
    average_aum_cr: float | None = None
    scheme_type: str | None = None
    scheme_category_label: str | None = None
    scheme_nav_name: str | None = None
    master_launch_date: str | None = None
    master_match_type: str = "unmatched"
    nav_source: str = "missing"
    nav_confidence: str = "missing"
    nav_is_synthetic: bool = False
    nav_notes: str | None = None


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_int(value: str, default: int = 0) -> int:
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return default
    return int(float(cleaned))


def parse_float(value: str, default: float | None = 0.0) -> float | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return default
    return float(cleaned)


def parse_amount_text(value: str) -> int | None:
    match = re.search(r"(\d[\d,]*)", value or "")
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def canonicalize_amc_name(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if normalized in AMC_CANONICAL_MAP:
        return AMC_CANONICAL_MAP[normalized]

    generic = normalized
    for phrase in GENERIC_AMC_WORDS:
        generic = generic.replace(phrase, " ")

    return normalize_spaces(generic)


def normalize_scheme_name(text: str) -> str:
    normalized = f" {text.lower().replace('&', ' and ').replace('–', ' ').replace('-', ' ')} "
    for old, new in SCHEME_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)

    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(token for token in normalized.split() if token not in SCHEME_STOPWORDS)


def infer_plan_style(*texts: str | None) -> str:
    combined = " ".join(text.lower() for text in texts if text).strip()
    income_markers = ("idcw", "dividend", "payout", "reinvestment")

    if "direct" in combined and "growth" in combined:
        return "direct_growth"
    if "growth" in combined:
        return "growth"
    if "direct" in combined and any(marker in combined for marker in income_markers):
        return "direct_income"
    if any(marker in combined for marker in income_markers):
        return "income"
    return "neutral"


def parse_archive_date_key(path: Path) -> tuple[int, int, int]:
    _, year, month, day = path.stem.split("_")
    return int(year), int(month), int(day)


@lru_cache(maxsize=1)
def latest_archive_snapshot_path(archive_dir: str = str(ARCHIVE_NAV_DIR)) -> str | None:
    files = sorted(Path(archive_dir).glob("*.csv"), key=parse_archive_date_key)
    if not files:
        return None
    return str(files[-1])


@lru_cache(maxsize=2)
def load_archive_nav_records(archive_dir: str = str(ARCHIVE_NAV_DIR)) -> tuple[ArchiveNavRecord, ...]:
    latest_path = latest_archive_snapshot_path(archive_dir)
    if latest_path is None:
        return ()

    records: list[ArchiveNavRecord] = []
    with Path(latest_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            nav = parse_float(row["Net Asset Value"], default=None)
            if nav is None:
                continue

            records.append(
                ArchiveNavRecord(
                    scheme_code=row["Scheme Code"].strip(),
                    scheme_name=row["Scheme Name"].strip(),
                    nav=nav,
                    date=row["Date"].strip(),
                    normalized_scheme_name=normalize_scheme_name(row["Scheme Name"]),
                    plan_style=infer_plan_style(row["Scheme Name"]),
                )
            )

    return tuple(records)


@lru_cache(maxsize=2)
def load_master_scheme_records(dataset_path: str = str(MASTER_DATASET_PATH)) -> tuple[MasterSchemeRecord, ...]:
    path = Path(dataset_path)
    if not path.exists():
        return ()

    records: list[MasterSchemeRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                MasterSchemeRecord(
                    scheme_code=row["Scheme_Code"].strip(),
                    scheme_name=row["Scheme_Name"].strip(),
                    amc=row["AMC"].strip(),
                    scheme_type=row["Scheme_Type"].strip(),
                    scheme_category=row["Scheme_Category"].strip(),
                    scheme_nav_name=row["Scheme_NAV_Name"].strip(),
                    scheme_min_amount=parse_amount_text(row["Scheme_Min_Amt"]),
                    nav=parse_float(row["NAV"], default=None),
                    latest_nav_date=row["Latest_NAV_Date"].strip() or None,
                    average_aum_cr=parse_float(row["Average_AUM_Cr"], default=None),
                    launch_date=row["Launch_Date"].strip() or None,
                    closure_date=row["Closure_Date"].strip() or None,
                    canonical_amc_name=canonicalize_amc_name(row["AMC"]),
                    normalized_scheme_name=normalize_scheme_name(row["Scheme_Name"]),
                    plan_style=infer_plan_style(row["Scheme_NAV_Name"], row["Scheme_Name"]),
                )
            )

    return tuple(records)


def build_master_index(records: tuple[MasterSchemeRecord, ...]) -> dict[str, list[MasterSchemeRecord]]:
    index: dict[str, list[MasterSchemeRecord]] = defaultdict(list)
    for record in records:
        index[record.normalized_scheme_name].append(record)
    return index


def build_archive_indices(
    records: tuple[ArchiveNavRecord, ...],
) -> tuple[dict[str, list[ArchiveNavRecord]], dict[str, list[ArchiveNavRecord]]]:
    by_code: dict[str, list[ArchiveNavRecord]] = defaultdict(list)
    by_name: dict[str, list[ArchiveNavRecord]] = defaultdict(list)

    for record in records:
        by_name[record.normalized_scheme_name].append(record)
        if record.scheme_code:
            by_code[record.scheme_code].append(record)

    return by_code, by_name


def master_record_score(record: MasterSchemeRecord, target_style: str) -> tuple[int, int, float, float]:
    style_match = int(record.plan_style == target_style)
    nav_hint = int(record.nav is not None and record.latest_nav_date is not None)
    aum_hint = record.average_aum_cr or -1.0
    nav_value = record.nav or -1.0
    return (style_match, nav_hint, aum_hint, nav_value)


def archive_record_score(record: ArchiveNavRecord, target_style: str) -> tuple[int, int, float]:
    style_match = int(record.plan_style == target_style)
    growth_preference = int(record.plan_style in {"direct_growth", "growth"})
    return (style_match, growth_preference, record.nav)


def select_best_master_record(
    candidates: list[MasterSchemeRecord],
    target_style: str,
) -> MasterSchemeRecord:
    return max(candidates, key=lambda item: master_record_score(item, target_style))


def select_best_archive_record(
    candidates: list[ArchiveNavRecord],
    target_style: str,
) -> ArchiveNavRecord:
    return max(candidates, key=lambda item: archive_record_score(item, target_style))


def match_master_record(
    scheme_name: str,
    amc_name: str,
    master_index: dict[str, list[MasterSchemeRecord]],
) -> tuple[MasterSchemeRecord | None, str]:
    normalized_name = normalize_scheme_name(scheme_name)
    candidates = master_index.get(normalized_name, [])
    if not candidates:
        return None, "unmatched"

    target_amc = canonicalize_amc_name(amc_name)
    target_style = infer_plan_style(scheme_name)
    same_amc_candidates = [item for item in candidates if item.canonical_amc_name == target_amc]
    if same_amc_candidates:
        return select_best_master_record(same_amc_candidates, target_style), "exact_name_amc"

    if len(candidates) == 1:
        return candidates[0], "scheme_name_only"

    candidate_amcs = {item.canonical_amc_name for item in candidates}
    if len(candidate_amcs) == 1:
        return select_best_master_record(candidates, target_style), "scheme_name_only"

    return None, "ambiguous"


def resolve_real_nav(
    scheme_name: str,
    master_record: MasterSchemeRecord | None,
    master_match_type: str,
    archive_by_code: dict[str, list[ArchiveNavRecord]],
    archive_by_name: dict[str, list[ArchiveNavRecord]],
) -> tuple[float | None, str | None, str, str, str | None]:
    target_style = infer_plan_style(scheme_name, master_record.scheme_nav_name if master_record else None)

    if master_record is not None and master_record.nav is not None:
        confidence = "high" if master_match_type == "exact_name_amc" else "medium"
        return (
            master_record.nav,
            master_record.latest_nav_date,
            "master",
            confidence,
            f"Master dataset via {master_match_type}.",
        )

    if master_record is not None and master_record.scheme_code in archive_by_code:
        archive_record = select_best_archive_record(archive_by_code[master_record.scheme_code], target_style)
        return (
            archive_record.nav,
            archive_record.date,
            "archive_code",
            "medium",
            "Archive NAV matched using scheme code.",
        )

    normalized_name = normalize_scheme_name(scheme_name)
    if normalized_name in archive_by_name:
        archive_record = select_best_archive_record(archive_by_name[normalized_name], target_style)
        return (
            archive_record.nav,
            archive_record.date,
            "archive_name",
            "medium",
            "Archive NAV matched using normalized scheme name.",
        )

    return None, None, "missing", "missing", None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * fraction)), 0), len(ordered) - 1)
    return ordered[index]


def select_peer_pool(
    scheme: FundScheme,
    real_schemes: tuple[FundScheme, ...],
) -> tuple[list[FundScheme], str]:
    target_style = infer_plan_style(scheme.scheme_name, scheme.scheme_nav_name)

    style_subcategory = [
        item
        for item in real_schemes
        if item.sub_category == scheme.sub_category
        and infer_plan_style(item.scheme_name, item.scheme_nav_name) == target_style
    ]
    if len(style_subcategory) >= 5:
        return style_subcategory, "imputed_subcategory_style"

    subcategory = [item for item in real_schemes if item.sub_category == scheme.sub_category]
    if len(subcategory) >= 5:
        return subcategory, "imputed_subcategory"

    category_risk = [
        item
        for item in real_schemes
        if item.category == scheme.category and item.risk_level == scheme.risk_level
    ]
    if len(category_risk) >= 8:
        return category_risk, "imputed_category_risk"

    category = [item for item in real_schemes if item.category == scheme.category]
    if len(category) >= 10:
        return category, "imputed_category"

    return list(real_schemes), "imputed_global"


def impute_nav_from_peers(scheme: FundScheme, peers: list[FundScheme]) -> float:
    peer_navs = [item.nav for item in peers if item.nav is not None and not item.nav_is_synthetic]
    if not peer_navs:
        return 10.0

    peer_ages = [item.fund_age_yr for item in peers if item.fund_age_yr > 0]
    peer_returns_3yr = [item.returns_3yr for item in peers if item.returns_3yr is not None]
    peer_returns_5yr = [item.returns_5yr for item in peers if item.returns_5yr is not None]

    base_nav = median(peer_navs)
    age_baseline = median(peer_ages) if peer_ages else max(scheme.fund_age_yr, 1.0)
    age_factor = clamp((max(scheme.fund_age_yr, 1.0) / max(age_baseline, 1.0)) ** 0.18, 0.78, 1.40)

    return_factor = 1.0
    if scheme.returns_3yr is not None and peer_returns_3yr:
        return_factor *= clamp(
            1 + ((scheme.returns_3yr - median(peer_returns_3yr)) / 100.0) * 0.40,
            0.86,
            1.18,
        )
    if scheme.returns_5yr is not None and peer_returns_5yr:
        return_factor *= clamp(
            1 + ((scheme.returns_5yr - median(peer_returns_5yr)) / 100.0) * 0.30,
            0.86,
            1.18,
        )

    estimated = base_nav * age_factor * return_factor
    lower_bound = max(5.0, percentile(peer_navs, 0.10) * 0.75)
    upper_bound = max(lower_bound + 1.0, percentile(peer_navs, 0.90) * 1.25)
    estimated = clamp(estimated, lower_bound, upper_bound)
    return round(estimated, 4)


def fill_missing_navs(catalog: tuple[FundScheme, ...]) -> tuple[FundScheme, ...]:
    real_schemes = tuple(item for item in catalog if item.nav is not None and not item.nav_is_synthetic)
    completed: list[FundScheme] = []

    for scheme in catalog:
        if scheme.nav is not None:
            completed.append(scheme)
            continue

        peers, source = select_peer_pool(scheme, real_schemes)
        imputed_nav = impute_nav_from_peers(scheme, peers)
        completed.append(
            replace(
                scheme,
                nav=imputed_nav,
                latest_nav_date=None,
                nav_source=source,
                nav_confidence="synthetic",
                nav_is_synthetic=True,
                nav_notes=f"Synthetic peer-based NAV using {len(peers)} comparable schemes.",
            )
        )

    return tuple(completed)


@lru_cache(maxsize=4)
def load_fund_schemes(
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    master_dataset_path: str = str(MASTER_DATASET_PATH),
    archive_dir: str = str(ARCHIVE_NAV_DIR),
) -> tuple[FundScheme, ...]:
    path = Path(dataset_path)
    if not path.exists():
        return ()

    master_records = load_master_scheme_records(master_dataset_path)
    master_index = build_master_index(master_records)
    archive_records = load_archive_nav_records(archive_dir)
    archive_by_code, archive_by_name = build_archive_indices(archive_records)

    schemes: list[FundScheme] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            master_record, match_type = match_master_record(
                row["scheme_name"].strip(),
                row["amc_name"].strip(),
                master_index,
            )

            nav, nav_date, nav_source, nav_confidence, nav_notes = resolve_real_nav(
                row["scheme_name"].strip(),
                master_record,
                match_type,
                archive_by_code,
                archive_by_name,
            )

            schemes.append(
                FundScheme(
                    scheme_name=row["scheme_name"].strip(),
                    amc_name=row["amc_name"].strip(),
                    category=row["category"].strip(),
                    sub_category=row["sub_category"].strip(),
                    risk_level=parse_int(row["risk_level"]),
                    rating=parse_int(row["rating"]),
                    min_sip=parse_int(row["min_sip"]),
                    min_lumpsum=parse_int(row["min_lumpsum"]),
                    expense_ratio=parse_float(row["expense_ratio"], default=0.0) or 0.0,
                    fund_size_cr=parse_float(row["fund_size_cr"], default=0.0) or 0.0,
                    fund_age_yr=parse_float(row["fund_age_yr"], default=0.0) or 0.0,
                    returns_1yr=parse_float(row["returns_1yr"], default=None),
                    returns_3yr=parse_float(row["returns_3yr"], default=None),
                    returns_5yr=parse_float(row["returns_5yr"], default=None),
                    scheme_code=master_record.scheme_code if master_record else None,
                    nav=nav,
                    latest_nav_date=nav_date,
                    average_aum_cr=master_record.average_aum_cr if master_record else None,
                    scheme_type=master_record.scheme_type if master_record else None,
                    scheme_category_label=master_record.scheme_category if master_record else None,
                    scheme_nav_name=master_record.scheme_nav_name if master_record else None,
                    master_launch_date=master_record.launch_date if master_record else None,
                    master_match_type=match_type,
                    nav_source=nav_source,
                    nav_confidence=nav_confidence,
                    nav_is_synthetic=False,
                    nav_notes=nav_notes,
                )
            )

    return fill_missing_navs(tuple(schemes))


def dataset_summary(schemes: tuple[FundScheme, ...] | None = None) -> dict[str, object]:
    catalog = schemes if schemes is not None else load_fund_schemes()
    category_counts = Counter(item.category for item in catalog)
    risk_counts = Counter(item.risk_level for item in catalog)
    subcategory_counts = Counter(item.sub_category for item in catalog)
    amc_names = {item.amc_name for item in catalog}
    match_counts = Counter(item.master_match_type for item in catalog)
    nav_source_counts = Counter(item.nav_source for item in catalog)

    enriched_count = sum(1 for item in catalog if item.master_match_type != "unmatched")
    real_nav_count = sum(1 for item in catalog if item.nav is not None and not item.nav_is_synthetic)
    synthetic_nav_count = sum(1 for item in catalog if item.nav_is_synthetic)
    nav_count = sum(1 for item in catalog if item.nav is not None)
    aum_count = sum(1 for item in catalog if item.average_aum_cr is not None)

    return {
        "scheme_count": len(catalog),
        "amc_count": len(amc_names),
        "category_count": len(category_counts),
        "subcategory_count": len(subcategory_counts),
        "category_distribution": category_counts.most_common(),
        "risk_distribution": sorted(risk_counts.items()),
        "top_subcategories": subcategory_counts.most_common(8),
        "enriched_count": enriched_count,
        "enrichment_ratio": (enriched_count / len(catalog)) if catalog else 0.0,
        "nav_count": nav_count,
        "real_nav_count": real_nav_count,
        "synthetic_nav_count": synthetic_nav_count,
        "aum_count": aum_count,
        "master_match_distribution": match_counts.most_common(),
        "nav_source_distribution": nav_source_counts.most_common(),
    }


def export_canonical_fund_dataset(
    schemes: tuple[FundScheme, ...] | None = None,
    output_path: str = str(PROCESSED_DATASET_PATH),
) -> str:
    catalog = schemes if schemes is not None else load_fund_schemes()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(item) for item in catalog]
    fieldnames = list(rows[0].keys()) if rows else []
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(target)
