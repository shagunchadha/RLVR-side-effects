"""Import shim: tests/test_parser.py (unchanged copy of items_draft/test_parser.py) does
`from answer_parser import parse_generation`; the real module lives in items/."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "_items_answer_parser", pathlib.Path(__file__).resolve().parent.parent / "items" / "answer_parser.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_generation = _mod.parse_generation
