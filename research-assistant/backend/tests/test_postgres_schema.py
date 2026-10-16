from sqlalchemy import create_mock_engine

from app.db import Base


def test_schema_ddl_compiles_with_postgres_dialect():
    statements = []
    engine = create_mock_engine(
        "postgresql+psycopg2://",
        lambda statement, *args, **kwargs: statements.append(str(statement.compile(dialect=engine.dialect))),
    )

    Base.metadata.create_all(engine)

    ddl = "\n".join(statements)
    assert len(Base.metadata.tables) >= 16
    assert "CREATE TABLE users" in ddl
    assert "CREATE TABLE papers" in ddl
    assert "CREATE TABLE paper_fields" in ddl
    assert "CREATE TABLE research_gaps" in ddl
    assert "CREATE TABLE gap_evidence" in ddl
