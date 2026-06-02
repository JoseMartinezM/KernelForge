import json

from kernelforge.benchmark.llm_inference import _redacted_generation_for_log, write_run_manifest


def test_write_run_manifest_without_grammar(tmp_path):
    manifest_path = write_run_manifest(
        tmp_path / "run.jsonl",
        [{"model": "m", "provider": "p"}],
        generation={"max_tokens": 1},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["grammars"] == []
    assert manifest["generation"] == {"max_tokens": 1}


def test_write_run_manifest_snapshots_guided_grammar(tmp_path):
    grammar = 'root ::= "x"\n'
    grammar_path = tmp_path / "triton.gbnf"
    grammar_path.write_text(grammar, encoding="utf-8")

    manifest_path = write_run_manifest(
        tmp_path / "run.jsonl",
        [{"model": "m", "provider": "p"}],
        generation={
            "max_tokens": 1,
            "extra_body": {
                "guided_grammar": grammar,
                "guided_decoding_backend": "xgrammar",
            },
        },
        grammar_source_path=grammar_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["grammars"][0]["location"] == "generation.extra_body.guided_grammar"
    assert manifest["grammars"][0]["content"] == grammar
    assert manifest["generation"]["extra_body"]["guided_grammar"] == {
        "manifest_ref": "grammars[0]",
        "sha256": "31e662fe97eb4be80591a86ad358192894782faefdf2d20583aef2db33e94f88",
        "bytes": len(grammar.encode("utf-8")),
    }
    assert manifest["generation"]["extra_body"]["guided_decoding_backend"] == "xgrammar"


def test_redacted_generation_for_log_keeps_guided_grammar_out_of_progress_output():
    grammar = 'root ::= "x"\n'
    generation = {"max_tokens": 1, "extra_body": {"guided_grammar": grammar}}

    redacted = _redacted_generation_for_log(generation)

    assert redacted["extra_body"]["guided_grammar"] == {
        "redacted": True,
        "sha256": "31e662fe97eb4be80591a86ad358192894782faefdf2d20583aef2db33e94f88",
        "bytes": len(grammar.encode("utf-8")),
    }
    assert generation["extra_body"]["guided_grammar"] == grammar
