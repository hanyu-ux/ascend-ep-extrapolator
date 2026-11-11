"""HTML navigation stays lossless without copying the report per target."""
from html.parser import HTMLParser
import unittest
from unittest import mock

from test_ep128_v2_excel import renderer


def reverse_insertions(document, insertions):
    """Small-fixture reference for the original whole-document algorithm."""
    for position, text in reversed(insertions):
        document = document[:position] + text + document[position:]
    return document


class SliceTracked(str):
    def __new__(cls, text):
        obj = super().__new__(cls, text)
        obj.copied_characters = 0
        obj.slice_count = 0
        return obj

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(key, slice):
            self.copied_characters += len(value)
            self.slice_count += 1
        return value


class Targets(HTMLParser):
    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.ids, self.links = [], []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.links.append(attrs["href"][1:])


class HTMLPerformanceTests(unittest.TestCase):
    def test_single_pass_insertions_preserve_original_coordinates_and_order(self):
        source = "甲<&>\n🧪xyz"
        insertions = [(0, "头"), (2, "[A]"), (2, "[B]"), (len(source), "尾")]
        self.assertEqual(renderer._html_insertions(source, insertions),
                         reverse_insertions(source, insertions))
        self.assertEqual(renderer._html_insertions(source, []), source)

    def test_many_targets_copy_each_source_character_only_once(self):
        # Count characters copied, not wall-clock time or machine speed. This
        # catches the prior N-document-copies behavior without a flaky timer.
        source = SliceTracked(("图表与源数据 🧪\n" * 2000) + "结束")
        positions = list(range(0, len(source), 11))
        insertions = [(position, '<a id="n' + str(index) + '"></a>')
                      for index, position in enumerate(positions)]
        actual = renderer._html_insertions(source, insertions)
        self.assertEqual(source.copied_characters, len(source))
        self.assertEqual(source.slice_count, len(insertions) + 1)
        self.assertEqual(len(actual), len(source) + sum(len(text) for _, text in insertions))
        self.assertTrue(actual.endswith("结束"))

    def test_navigation_is_identical_to_reference_for_many_targets_and_escaping(self):
        blocks = []
        for index in range(80):
            blocks.append('<h2>case ' + str(index) + ' &amp; &lt;图&gt;</h2>\n'
                '<h3>函数 \' &quot; &amp;lt;script&amp;gt;</h3>'
                '<table><tr><th>源值</th><th>耗时</th></tr><tr><td>7.125</td><td>2.5</td></tr></table>'
                '<svg><title>真实曲线</title><polyline points="0,4.25 12,0 19,7.125"/>'
                '<circle><title>点提示不进目录</title></circle></svg>'
                '<div class="u-bars" role="img" aria-label="耗时 &amp; 范围">'
                '<div class="u-bar"><span>1.25 ms</span></div></div>')
        source = ('<html><body><h1>报告</h1><p id="report-nav-target-1">保留</p>'
                  '<div id="report-chart-directory">现有锚点</div>' + ''.join(blocks) + '</body></html>')
        actual = renderer._html_navigation(source)
        with mock.patch.object(renderer, "_html_insertions", side_effect=reverse_insertions):
            expected = renderer._html_navigation(source)
        self.assertEqual(actual, expected)
        parsed = Targets(actual)
        self.assertEqual(len(parsed.ids), len(set(parsed.ids)))
        self.assertTrue(set(parsed.links).issubset(set(parsed.ids)))
        self.assertEqual(actual.count('<polyline points="0,4.25 12,0 19,7.125"/>'), 80)
        self.assertEqual(actual.count('<td>7.125</td>'), 80)
        self.assertEqual(actual.count('<title>点提示不进目录</title>'), 80)
        self.assertEqual(actual.count('<script>'), 1)

    def test_navigation_assembles_original_document_once_after_indexing(self):
        source = ('<html><body><h1>报告</h1><h2>汇总</h2><table><tr><th>值</th>'
                  '</tr><tr><td>1.25</td></tr></table><svg><title>曲线</title></svg></body></html>')
        calls = []
        original = renderer._html_insertions

        def checked(document, insertions):
            self.assertIs(document, source)
            positions = [position for position, _ in insertions]
            self.assertEqual(positions, sorted(positions))
            self.assertTrue(all(0 <= position <= len(source) for position in positions))
            calls.append(len(insertions))
            return original(document, insertions)

        with mock.patch.object(renderer, "_html_insertions", side_effect=checked):
            renderer._html_navigation(source)
        self.assertEqual(calls, [4])  # Heading, table, SVG and directory.

    def test_missing_report_heading_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "主标题"):
            renderer._html_navigation('<html><h2>没有主标题</h2></html>')


if __name__ == "__main__":
    unittest.main()
