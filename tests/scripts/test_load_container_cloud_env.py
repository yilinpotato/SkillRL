from pathlib import Path

from scripts.load_container_cloud_env import parse_dotenv


def test_parse_dotenv_strips_shell_quotes_and_ignores_unrelated_keys(tmp_path: Path):
    dotenv = tmp_path / "cloud.env"
    dotenv.write_text(
        "DEEPSEEK_API_KEY='quoted-key'\n"
        "DEEPSEEK_MODEL=deepseek-v4-flash # default\n"
        "UNRELATED_SHELL_SETTING=do-not-export\n",
        encoding="utf-8",
    )

    assert parse_dotenv(dotenv) == {
        "DEEPSEEK_API_KEY": "quoted-key",
        "DEEPSEEK_MODEL": "deepseek-v4-flash",
    }


def test_parse_dotenv_rejects_unterminated_quote(tmp_path: Path):
    dotenv = tmp_path / "bad.env"
    dotenv.write_text("DEEPSEEK_API_KEY='unterminated\n", encoding="utf-8")

    try:
        parse_dotenv(dotenv)
    except ValueError as exc:
        assert "unterminated quoted value" in str(exc)
    else:
        raise AssertionError("expected malformed dotenv to be rejected")
