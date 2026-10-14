"""
Day 15 — custom eval catalog.

18 self-authored coding tasks against evals/tasks/toy_repo.
Pass/fail is mechanical: pytest on the listed files, plus optional
file_contains checks. Live scoring is Day 16 (evals/runner.py).
"""

from typing import TypedDict


class FileContains(TypedDict):
    path: str
    needle: str


class Task(TypedDict):
    id: str
    tier: str
    prompt: str
    pytest: list[str]
    file_contains: list[FileContains]
    max_llm_calls: int


TASKS: list[Task] = [
    {
        "id": "001-sum-of-squares",
        "tier": "easy",
        "prompt": (
            "Add a function sum_of_squares(nums) to toybox/mathutil.py that "
            "returns the sum of squares of the numbers in the list. Empty list "
            "is 0. Do not change other files. tests/test_001_sum_of_squares.py "
            "must pass."
        ),
        "pytest": ["tests/test_001_sum_of_squares.py"],
        "file_contains": [
            {"path": "toybox/mathutil.py", "needle": "def sum_of_squares"}
        ],
        "max_llm_calls": 16,
    },
    {
        "id": "002-is-anagram",
        "tier": "easy",
        "prompt": (
            "Add is_anagram(a, b) to toybox/textutil.py. Ignore case and spaces. "
            "tests/test_002_is_anagram.py must pass."
        ),
        "pytest": ["tests/test_002_is_anagram.py"],
        "file_contains": [{"path": "toybox/textutil.py", "needle": "def is_anagram"}],
        "max_llm_calls": 16,
    },
    {
        "id": "003-clamp",
        "tier": "easy",
        "prompt": (
            "Add clamp(n, lo, hi) to toybox/mathutil.py that returns n limited "
            "to [lo, hi]. tests/test_003_clamp.py must pass."
        ),
        "pytest": ["tests/test_003_clamp.py"],
        "file_contains": [{"path": "toybox/mathutil.py", "needle": "def clamp"}],
        "max_llm_calls": 16,
    },
    {
        "id": "004-fizzbuzz-counts",
        "tier": "easy",
        "prompt": (
            "Add fizzbuzz_counts(n) to toybox/textutil.py. For i in 1..n, count "
            "how many values are Fizz (divisible by 3 but not 5), Buzz "
            "(divisible by 5 but not 3), and FizzBuzz (divisible by both). "
            "Return a tuple (fizz, buzz, fizzbuzz). "
            "tests/test_004_fizzbuzz.py must pass."
        ),
        "pytest": ["tests/test_004_fizzbuzz.py"],
        "file_contains": [
            {"path": "toybox/textutil.py", "needle": "def fizzbuzz_counts"}
        ],
        "max_llm_calls": 16,
    },
    {
        "id": "005-inclusive-range",
        "tier": "easy",
        "prompt": (
            "toybox/mathutil.py has inclusive_range(start, end). It is missing "
            "the end value. Fix it so the range is inclusive. "
            "tests/test_005_inclusive_range.py must pass."
        ),
        "pytest": ["tests/test_005_inclusive_range.py"],
        "file_contains": [
            {"path": "toybox/mathutil.py", "needle": "def inclusive_range"}
        ],
        "max_llm_calls": 16,
    },
    {
        "id": "006-palindrome",
        "tier": "easy",
        "prompt": (
            "palindrome(s) in toybox/mathutil.py is wrong: it compares s to "
            "itself. It must return True iff s equals its reverse. "
            "tests/test_006_palindrome.py must pass."
        ),
        "pytest": ["tests/test_006_palindrome.py"],
        "file_contains": [{"path": "toybox/mathutil.py", "needle": "def palindrome"}],
        "max_llm_calls": 16,
    },
    {
        "id": "007-safe-div",
        "tier": "medium",
        "prompt": (
            "safe_div(a, b) in toybox/mathutil.py raises on b == 0. It must "
            "return None instead. tests/test_007_safe_div.py must pass."
        ),
        "pytest": ["tests/test_007_safe_div.py"],
        "file_contains": [{"path": "toybox/mathutil.py", "needle": "def safe_div"}],
        "max_llm_calls": 16,
    },
    {
        "id": "008-squish",
        "tier": "medium",
        "prompt": (
            "squish(s) in toybox/textutil.py collapses whitespace but does not "
            "strip the ends. Fix it. tests/test_008_squish.py must pass."
        ),
        "pytest": ["tests/test_008_squish.py"],
        "file_contains": [{"path": "toybox/textutil.py", "needle": "def squish"}],
        "max_llm_calls": 16,
    },
    {
        "id": "009-mean-empty",
        "tier": "medium",
        "prompt": (
            "mean(xs) in toybox/stats.py raises on an empty list. It must "
            "return None. tests/test_009_mean_empty.py must pass."
        ),
        "pytest": ["tests/test_009_mean_empty.py"],
        "file_contains": [{"path": "toybox/stats.py", "needle": "def mean"}],
        "max_llm_calls": 16,
    },
    {
        "id": "010-config-missing",
        "tier": "medium",
        "prompt": (
            "Read toybox/config.py. get(key) raises KeyError for unknown keys. "
            "Unknown keys must return None. tests/test_010_config_missing.py "
            "must pass."
        ),
        "pytest": ["tests/test_010_config_missing.py"],
        "file_contains": [{"path": "toybox/config.py", "needle": "def get"}],
        "max_llm_calls": 16,
    },
    {
        "id": "011-csv-header",
        "tier": "medium",
        "prompt": (
            "parse_rows in toybox/csvparse.py treats the header as data. Skip "
            "the first line. tests/test_011_csv_header.py must pass."
        ),
        "pytest": ["tests/test_011_csv_header.py"],
        "file_contains": [{"path": "toybox/csvparse.py", "needle": "def parse_rows"}],
        "max_llm_calls": 16,
    },
    {
        "id": "012-title-case",
        "tier": "medium",
        "prompt": (
            "title_case in toybox/textutil.py currently uppercases the whole "
            "string. It must capitalize each whitespace-separated word "
            "(e.g. 'hello world' -> 'Hello World'). "
            "tests/test_012_title_case.py must pass."
        ),
        "pytest": ["tests/test_012_title_case.py"],
        "file_contains": [{"path": "toybox/textutil.py", "needle": "def title_case"}],
        "max_llm_calls": 16,
    },
    {
        "id": "013-median-copy",
        "tier": "medium",
        "prompt": (
            "median in toybox/stats.py sorts the input in place. It must not "
            "mutate the caller's list. tests/test_013_median_copy.py must pass."
        ),
        "pytest": ["tests/test_013_median_copy.py"],
        "file_contains": [{"path": "toybox/stats.py", "needle": "def median"}],
        "max_llm_calls": 16,
    },
    {
        "id": "014-parser-and-caller",
        "tier": "hard",
        "prompt": (
            "parse_int_list in toybox/parser.py returns strings, so "
            "sum_parsed in toybox/caller.py cannot sum them. Make "
            "parse_int_list return integers (skip empty pieces). Fix caller.py "
            "too if needed. tests/test_014_parser_caller.py must pass."
        ),
        "pytest": ["tests/test_014_parser_caller.py"],
        "file_contains": [
            {"path": "toybox/parser.py", "needle": "def parse_int_list"}
        ],
        "max_llm_calls": 24,
    },
    {
        "id": "015-moving-average",
        "tier": "hard",
        "prompt": (
            "Add moving_average(xs, window) to toybox/stats.py. Return the list "
            "of means of each contiguous window of `window` items. "
            "You may use mean(). tests/test_015_moving_average.py must pass."
        ),
        "pytest": ["tests/test_015_moving_average.py"],
        "file_contains": [{"path": "toybox/stats.py", "needle": "def moving_average"}],
        "max_llm_calls": 24,
    },
    {
        "id": "016-config-env",
        "tier": "hard",
        "prompt": (
            "get_with_env(key) in toybox/config.py should honor TOYBOX_{KEY} "
            "in the environment (uppercase key). When set, parse it as int if "
            "the default is an int, else keep it as str. When unset, return "
            "the default. tests/test_016_config_env.py must pass."
        ),
        "pytest": ["tests/test_016_config_env.py"],
        "file_contains": [{"path": "toybox/config.py", "needle": "def get_with_env"}],
        "max_llm_calls": 24,
    },
    {
        "id": "017-wrong-file-named",
        "tier": "adversarial",
        "prompt": (
            "Implement factorial(n) for n >= 0. The notes in toybox/legacy.py "
            "say leftover helpers belong there. tests/test_017_factorial.py "
            "must pass."
        ),
        "pytest": ["tests/test_017_factorial.py"],
        "file_contains": [{"path": "toybox/mathutil.py", "needle": "def factorial"}],
        "max_llm_calls": 20,
    },
    {
        "id": "018-wrong-new-module",
        "tier": "adversarial",
        "prompt": (
            "Create toybox/extra.py and put def doubled(x) there (return 2*x). "
            "Wire it however you like. tests/test_018_doubled.py must pass."
        ),
        "pytest": ["tests/test_018_doubled.py"],
        "file_contains": [{"path": "toybox/mathutil.py", "needle": "def doubled"}],
        "max_llm_calls": 20,
    },
]


def task_by_id(task_id: str) -> Task:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)
