from SEJO_SDK import cli


def test_cli_help_returns_success(capsys):
    assert cli.main([]) == 0

    output = capsys.readouterr().out

    assert "SEJO SDK CLI" in output


def test_cli_doctor_reports_environment(capsys):
    assert cli.main(["doctor"]) == 0

    output = capsys.readouterr().out

    assert "SEJO SDK Doctor" in output
    assert "Optional extras:" in output
    assert "Development tools:" in output


def test_cli_doctor_strict_returns_failure_when_checks_missing(monkeypatch):
    monkeypatch.setattr(cli, "OPTIONAL_DEPENDENCIES", {"missing": "missing_module"})
    monkeypatch.setattr(cli, "DEV_TOOLS", {})

    assert cli.main(["doctor", "--strict"]) == 1


def test_cli_doctor_strict_returns_success_when_checks_pass(monkeypatch):
    monkeypatch.setattr(cli, "OPTIONAL_DEPENDENCIES", {"sys": "sys"})
    monkeypatch.setattr(cli, "DEV_TOOLS", {})

    assert cli.main(["doctor", "--strict"]) == 0
