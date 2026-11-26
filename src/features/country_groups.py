"""Single source of truth for the Country grouping (4 groups).

Imported by stage3b_group_and_fix_skew.py, clustering_pipeline.py and the figure script,
so training data and new bookings (inference) are always mapped identically.

Groups:
  Portugal        PRT (the hotel is in Lisbon: domestic guests)
  Latin Europe    FRA, ESP, ITA (Portugal's nearest Latin-language neighbours)
  Rest of Europe  every other European code
  Outside Europe  Asia, the Americas, Africa, Oceania, and unknown / junk codes
"""
PORTUGAL = ["PRT"]
LATIN_EUROPE = ["FRA", "ESP", "ITA"]
EUROPE = [
    "ALB", "AND", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CZE", "DEU", "DNK", "ESP", "EST",
    "FIN", "FRA", "FRO", "GBR", "GGY", "GIB", "GRC", "HRV", "HUN", "IMN", "IRL", "ISL", "ITA",
    "JEY", "LIE", "LTU", "LUX", "LVA", "MCO", "MKD", "MLT", "MNE", "NLD", "NOR", "POL", "PRT",
    "ROU", "RUS", "SRB", "SVK", "SVN", "SWE", "UKR",
]
GROUP_ORDER = ["Portugal", "Latin Europe", "Rest of Europe", "Outside Europe"]


def country_group(code):
    """Map a raw country code to one of the 4 groups. Unknown / empty / unseen codes -> Outside Europe."""
    code = "" if code is None else str(code).strip()
    if code == "PRT":
        return "Portugal"
    if code in LATIN_EUROPE:
        return "Latin Europe"
    if code in EUROPE:
        return "Rest of Europe"
    return "Outside Europe"
