from __future__ import annotations

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


# ---------------------------------------------------------------------------
# eval subcommand
# ---------------------------------------------------------------------------

def test_eval_missing_file():
    rc = cli.main(["eval", "/nonexistent/file.py"])
    assert rc == 1


def test_eval_file_without_suite(tmp_path, capsys):
    f = tmp_path / "no_suite.py"
    f.write_text("x = 1\n")
    rc = cli.main(["eval", str(f)])
    assert rc == 1


def test_eval_file_without_agent(tmp_path):
    f = tmp_path / "suite_only.py"
    f.write_text(
        "from SEJO_SDK.evals import EvalSuite, EvalCase, contains\n"
        "suite = EvalSuite(cases=[EvalCase('q','a')], scorer=contains)\n"
    )
    rc = cli.main(["eval", str(f)])
    assert rc == 1


def test_eval_file_runs_suite(tmp_path, capsys):
    f = tmp_path / "full_eval.py"
    f.write_text(
        "from SEJO_SDK.agent import Agent\n"
        "from SEJO_SDK.evals import EvalSuite, EvalCase, contains\n"
        "from SEJO_SDK.messages import Message, ModelResponse\n"
        "from SEJO_SDK.model import ModelClient\n"
        "class M(ModelClient):\n"
        "    def send_prompt(self, p, **_): return ModelResponse(content=p)\n"
        "    def send_messages(self, msgs, **_):\n"
        "        return ModelResponse(content=msgs[-1].content if msgs else '')\n"
        "    def stream_response(self, p, **_): yield p\n"
        "    def stream_messages(self, msgs, **_): yield ''\n"
        "agent = Agent(model=M())\n"
        "suite = EvalSuite(\n"
        "    cases=[EvalCase('hello world', expected='hello')],\n"
        "    scorer=contains,\n"
        ")\n"
    )
    rc = cli.main(["eval", str(f)])
    assert rc == 0


def test_eval_fail_under(tmp_path):
    f = tmp_path / "fail_eval.py"
    f.write_text(
        "from SEJO_SDK.agent import Agent\n"
        "from SEJO_SDK.evals import EvalSuite, EvalCase, contains\n"
        "from SEJO_SDK.messages import Message, ModelResponse\n"
        "from SEJO_SDK.model import ModelClient\n"
        "class M(ModelClient):\n"
        "    def send_prompt(self, p, **_): return ModelResponse(content='wrong')\n"
        "    def send_messages(self, msgs, **_):\n"
        "        return ModelResponse(content='wrong')\n"
        "    def stream_response(self, p, **_): yield 'wrong'\n"
        "    def stream_messages(self, msgs, **_): yield 'wrong'\n"
        "agent = Agent(model=M())\n"
        "suite = EvalSuite(\n"
        "    cases=[EvalCase('q', expected='NEVER_MATCHES')],\n"
        "    scorer=contains,\n"
        ")\n"
    )
    rc = cli.main(["eval", str(f), "--fail-under", "1.0"])
    assert rc == 1
