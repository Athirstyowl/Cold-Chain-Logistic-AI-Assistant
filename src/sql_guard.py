import re
from typing import Optional

# Defence in depth only: the USR_FDE_RO database login is the real boundary.
# This rejects anything that is not a single read-only statement before it
# reaches the database.

_FORBIDDEN = (
    "INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "EXEC", "EXECUTE", "GRANT", "REVOKE", "DENY", "INTO", "BACKUP", "RESTORE",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN) + r")\b", re.IGNORECASE)


def _strip_comments_and_literals(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_read_only_sql(sql: str) -> Optional[str]:
    """Return None if the query is a single SELECT/WITH statement, else the reason it is blocked."""
    cleaned = _strip_comments_and_literals(sql or "").strip().rstrip(";").strip()

    if not cleaned:
        return "Empty query."
    if ";" in cleaned:
        return "Only a single statement is allowed."
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return "Only SELECT queries are authorized on this view."
    forbidden = _FORBIDDEN_RE.search(cleaned)
    if forbidden:
        return f"Keyword '{forbidden.group(1).upper()}' is not allowed."
    return None
