"""Pure helpers of the HTTP layer: JSON safety, range parsing, tag filters, ETag matching."""

from __future__ import annotations

import pytest

from forge.api import SafeJSONResponse, etag_matches, error_response, etag_for, parse_range, parse_tag_filters
from forge.errors import ForgeError, ValidationFailed


def test_json_responses_replace_non_finite_numbers() -> None:
    body = SafeJSONResponse({"a": float("nan"), "b": [float("inf"), -float("inf"), 1.5], "c": {"d": float("nan")}}).body
    assert body == b'{"a":null,"b":[null,null,1.5],"c":{"d":null}}'


def test_error_envelope_shape() -> None:
    assert error_response(409, "conflict", "no room", {"need_gb": 5}).body == b'{"error":{"code":"conflict","message":"no room","details":{"need_gb":5}}}'
    assert error_response(404, "not_found", "gone").body == b'{"error":{"code":"not_found","message":"gone"}}'


@pytest.mark.parametrize(
    ("header", "size", "span"),
    [
        (None, 100, None),
        ("", 100, None),
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=10-", 100, (10, 99)),
        ("bytes=-10", 100, (90, 99)),
        ("bytes=90-500", 100, (90, 99)),
        ("bytes=-500", 100, (0, 99)),
        ("bytes=99-99", 100, (99, 99)),
        (" bytes=0-0 ", 100, (0, 0)),
        ("bytes=5-2", 100, None),
        ("bytes=0-1,4-5", 100, None),
        ("bytes=x-y", 100, None),
        ("bytes=-", 100, None),
        ("chunks=0-9", 100, None),
        ("bytes=" + "9" * 40 + "-", 100, "416"),
        ("bytes=100-", 100, "416"),
        ("bytes=-0", 100, "416"),
        ("bytes=0-0", 0, "416"),
    ],
)
def test_parse_range(header: str | None, size: int, span: tuple[int, int] | str | None) -> None:
    if span == "416":
        with pytest.raises(ForgeError) as caught:
            parse_range(header, size)
        assert caught.value.status == 416 and caught.value.details == {"size": size}
    else:
        assert parse_range(header, size) == span


def test_etag_matching() -> None:
    etag = etag_for("a_0123456789abcdef01234567")
    assert etag == '"a_0123456789abcdef01234567"'
    assert etag_matches(etag, etag) and etag_matches("W/" + etag, etag) and etag_matches('"x", ' + etag, etag) and etag_matches("*", etag)
    assert not etag_matches('"x"', etag) and not etag_matches("", etag)


def test_tag_filters() -> None:
    assert parse_tag_filters(['{"family": "synth", "mood": ["dark", "wide"]}', "family:bass", "k = v".replace(" = ", "=")]) == {
        "family": ["synth", "bass"], "mood": ["dark", "wide"], "k": ["v"],
    }
    assert parse_tag_filters([]) == {}
    for bad in ("nothing", ":v", "k:", "{oops", "[]", '{"a": 1}', '{"a": [1]}', "k:" + "v" * 201, "k" * 65 + ":v"):
        with pytest.raises(ValidationFailed) as caught:
            parse_tag_filters([bad])
        assert caught.value.errors[0]["path"] == "tags"
