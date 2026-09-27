import pytest

from src.sql_guard import validate_read_only_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP 5 * FROM FDE_VIEWS.VW_ACTIVE_FLEET",
        "  select Latitude, Longitude FROM FDE_VIEWS.VW_ACTIVE_FLEET;  ",
        "WITH hot AS (SELECT * FROM FDE_VIEWS.VW_ACTIVE_FLEET WHERE Current_Temperature_C > 8) SELECT COUNT(*) FROM hot",
        "-- recent readings\nSELECT TOP 1 * FROM FDE_VIEWS.VW_ACTIVE_FLEET",
        "SELECT * FROM FDE_VIEWS.VW_ACTIVE_FLEET WHERE Risk_Classification = 'High Risk; DROP'",
    ],
)
def test_read_only_queries_are_allowed(sql):
    assert validate_read_only_sql(sql) is None


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO FDE_VIEWS.AgentAuditLog (Content) VALUES ('x')",
        "DROP TABLE dbo.TBL_SC_FLEET_HIST_RAW",
        "SELECT 1; DROP TABLE dbo.TBL_SC_FLEET_HIST_RAW",
        "/* harmless */ DELETE FROM dbo.TBL_SC_FLEET_HIST_RAW",
        "SELECT * INTO dbo.copy FROM FDE_VIEWS.VW_ACTIVE_FLEET",
        "WITH x AS (SELECT 1 AS a) UPDATE dbo.TBL_SC_FLEET_HIST_RAW SET V_LAT = 0",
        "EXEC sp_configure",
        "",
        "   ",
    ],
)
def test_write_or_multi_statement_queries_are_blocked(sql):
    assert validate_read_only_sql(sql) is not None
