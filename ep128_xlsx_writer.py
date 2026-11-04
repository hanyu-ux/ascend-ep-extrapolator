#!/usr/bin/env python3
"""Small, dependency-free XLSX writer for the EP128 calculation workbook.

Rows and columns are one-based.  Formulas retain a supplied cached result so
readers can show the report before Excel recalculates the workbook.
"""
from datetime import datetime, timezone
import math
import numbers
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


WRITER_VERSION = "20261104.1.line_charts"
MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "http://schemas.openxmlformats.org/package/2006/content-types"
XML = "http://www.w3.org/XML/1998/namespace"
CHART = "http://schemas.openxmlformats.org/drawingml/2006/chart"
DRAWING = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
ART = "http://schemas.openxmlformats.org/drawingml/2006/main"
MAX_ROW, MAX_COL = 1048576, 16384
ET.register_namespace("", MAIN)
ET.register_namespace("r", REL)
ET.register_namespace("c", CHART)
ET.register_namespace("xdr", DRAWING)
ET.register_namespace("a", ART)


def _node(parent, name, attrs=None, text=None):
    child = ET.SubElement(parent, "{" + MAIN + "}" + name,
                          {key: str(value) for key, value in (attrs or {}).items()})
    if text is not None:
        child.text = str(text)
    return child


def _xml(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _drawing_node(parent, namespace, name, attrs=None, text=None):
    node = ET.SubElement(parent, "{" + namespace + "}" + name,
                         {key: str(value) for key, value in (attrs or {}).items()})
    if text is not None:
        node.text = str(text)
    return node


def _chart_node(parent, name, attrs=None, text=None):
    return _drawing_node(parent, CHART, name, attrs, text)


def _art_node(parent, name, attrs=None, text=None):
    return _drawing_node(parent, ART, name, attrs, text)


def _chart_title(parent, text, size=1400):
    title = _chart_node(parent, "title")
    rich = _chart_node(_chart_node(title, "tx"), "rich")
    _art_node(rich, "bodyPr")
    _art_node(rich, "lstStyle")
    paragraph = _art_node(rich, "p")
    props = _art_node(_art_node(paragraph, "pPr"), "defRPr", {"sz": size})
    _art_node(props, "latin", {"typeface": "Arial"})
    _art_node(props, "ea", {"typeface": "微软雅黑"})
    run = _art_node(paragraph, "r")
    _art_node(run, "t", text=text)
    _chart_node(title, "layout")
    _chart_node(title, "overlay", {"val": 0})


def _chart_data(parent, values, reference=None, numeric=False):
    """Write range caches, preserving sparse point indices for missing values."""
    if reference is None:
        data = _chart_node(parent, "numLit" if numeric else "strLit")
    else:
        ref = _chart_node(parent, "numRef" if numeric else "strRef")
        _chart_node(ref, "f", text=reference)
        data = _chart_node(ref, "numCache" if numeric else "strCache")
    if numeric:
        _chart_node(data, "formatCode", text="0.000")
    _chart_node(data, "ptCount", {"val": len(values)})
    for index, value in enumerate(values):
        if value is None:
            continue
        _chart_node(_chart_node(data, "pt", {"idx": index}), "v", text=value)


def _bar_chart_xml(spec):
    root = ET.Element("{" + CHART + "}chartSpace")
    _chart_node(root, "lang", {"val": "zh-CN"})
    _chart_node(root, "roundedCorners", {"val": 0})
    chart = _chart_node(root, "chart")
    _chart_title(chart, spec["title"])
    _chart_node(chart, "autoTitleDeleted", {"val": 0})
    plot = _chart_node(chart, "plotArea")
    _chart_node(plot, "layout")
    bar = _chart_node(plot, "barChart")
    _chart_node(bar, "barDir", {"val": "bar"})
    _chart_node(bar, "grouping", {"val": "stacked" if spec["stacked"] else "clustered"})
    _chart_node(bar, "varyColors", {"val": 0})
    for index, series in enumerate(spec["series"]):
        ser = _chart_node(bar, "ser")
        _chart_node(ser, "idx", {"val": index})
        _chart_node(ser, "order", {"val": index})
        tx = _chart_node(ser, "tx")
        if series.get("name_ref"):
            _chart_data(tx, [series["name"]], series["name_ref"])
        else:
            _chart_node(tx, "v", text=series["name"])
        shape = _chart_node(ser, "spPr")
        _art_node(_art_node(shape, "solidFill"), "srgbClr", {"val": series["color"]})
        _art_node(_art_node(shape, "ln"), "noFill")
        _chart_data(_chart_node(ser, "cat"), spec["categories"], spec["categories_ref"])
        _chart_data(_chart_node(ser, "val"), series["values"], series.get("values_ref"), numeric=True)
    _chart_node(bar, "gapWidth", {"val": 65})
    _chart_node(bar, "overlap", {"val": 100 if spec["stacked"] else 0})
    for number in (100, 200):
        _chart_node(bar, "axId", {"val": number})
    category = _chart_node(plot, "catAx")
    _chart_node(category, "axId", {"val": 100})
    _chart_node(_chart_node(category, "scaling"), "orientation", {"val": "maxMin"})
    _chart_node(category, "delete", {"val": 0})
    _chart_node(category, "axPos", {"val": "l"})
    _chart_node(category, "majorTickMark", {"val": "none"})
    _chart_node(category, "minorTickMark", {"val": "none"})
    _chart_node(category, "tickLblPos", {"val": "nextTo"})
    _chart_node(category, "crossAx", {"val": 200})
    _chart_node(category, "crosses", {"val": "autoZero"})
    _chart_node(category, "auto", {"val": 1})
    _chart_node(category, "lblAlgn", {"val": "ctr"})
    _chart_node(category, "lblOffset", {"val": 100})
    value = _chart_node(plot, "valAx")
    _chart_node(value, "axId", {"val": 200})
    _chart_node(_chart_node(value, "scaling"), "orientation", {"val": "minMax"})
    _chart_node(value, "delete", {"val": 0})
    _chart_node(value, "axPos", {"val": "b"})
    grid = _chart_node(value, "majorGridlines")
    _art_node(_art_node(_art_node(_chart_node(grid, "spPr"), "ln"), "solidFill"),
              "srgbClr", {"val": "DFE7EE"})
    if spec["x_axis_title"]:
        _chart_title(value, spec["x_axis_title"], size=1000)
    _chart_node(value, "numFmt", {"formatCode": "0.000", "sourceLinked": 0})
    _chart_node(value, "majorTickMark", {"val": "none"})
    _chart_node(value, "minorTickMark", {"val": "none"})
    _chart_node(value, "tickLblPos", {"val": "nextTo"})
    _chart_node(value, "crossAx", {"val": 100})
    _chart_node(value, "crosses", {"val": "max"})
    _chart_node(value, "crossBetween", {"val": "between"})
    legend = _chart_node(chart, "legend")
    _chart_node(legend, "legendPos", {"val": "b"})
    _chart_node(legend, "layout")
    _chart_node(legend, "overlay", {"val": 0})
    _chart_node(chart, "plotVisOnly", {"val": 1})
    _chart_node(chart, "dispBlanksAs", {"val": "gap"})
    _chart_node(chart, "showDLblsOverMax", {"val": 0})
    _art_node(_art_node(_chart_node(root, "spPr"), "solidFill"), "srgbClr", {"val": "FFFFFF"})
    return root


def _line_chart_xml(spec):
    root = ET.Element("{" + CHART + "}chartSpace")
    _chart_node(root, "lang", {"val": "zh-CN"})
    _chart_node(root, "roundedCorners", {"val": 0})
    chart = _chart_node(root, "chart")
    _chart_title(chart, spec["title"])
    _chart_node(chart, "autoTitleDeleted", {"val": 0})
    plot = _chart_node(chart, "plotArea")
    _chart_node(plot, "layout")
    line = _chart_node(plot, "lineChart")
    _chart_node(line, "grouping", {"val": "standard"})
    _chart_node(line, "varyColors", {"val": 0})
    for index, series in enumerate(spec["series"]):
        ser = _chart_node(line, "ser")
        _chart_node(ser, "idx", {"val": index})
        _chart_node(ser, "order", {"val": index})
        tx = _chart_node(ser, "tx")
        if series.get("name_ref"):
            _chart_data(tx, [series["name"]], series["name_ref"])
        else:
            _chart_node(tx, "v", text=series["name"])
        stroke = _art_node(_chart_node(ser, "spPr"), "ln", {"w": 25400})
        _art_node(_art_node(stroke, "solidFill"), "srgbClr", {"val": series["color"]})
        _art_node(stroke, "prstDash", {"val": "solid"})
        marker = _chart_node(ser, "marker")
        _chart_node(marker, "symbol", {"val": "circle"})
        _chart_node(marker, "size", {"val": 5})
        marker_shape = _chart_node(marker, "spPr")
        _art_node(_art_node(marker_shape, "solidFill"), "srgbClr", {"val": series["color"]})
        _art_node(_art_node(_art_node(marker_shape, "ln"), "solidFill"),
                  "srgbClr", {"val": series["color"]})
        _chart_data(_chart_node(ser, "cat"), spec["categories"], spec["categories_ref"])
        _chart_data(_chart_node(ser, "val"), series["values"], series.get("values_ref"), numeric=True)
        _chart_node(ser, "smooth", {"val": 0})
    _chart_node(line, "marker", {"val": 1})
    _chart_node(line, "smooth", {"val": 0})
    for number in (100, 200):
        _chart_node(line, "axId", {"val": number})
    category = _chart_node(plot, "catAx")
    _chart_node(category, "axId", {"val": 100})
    _chart_node(_chart_node(category, "scaling"), "orientation", {"val": "minMax"})
    _chart_node(category, "delete", {"val": 0})
    _chart_node(category, "axPos", {"val": "b"})
    if spec["x_axis_title"]:
        _chart_title(category, spec["x_axis_title"], size=1000)
    _chart_node(category, "majorTickMark", {"val": "none"})
    _chart_node(category, "minorTickMark", {"val": "none"})
    _chart_node(category, "tickLblPos", {"val": "nextTo"})
    _chart_node(category, "crossAx", {"val": 200})
    _chart_node(category, "crosses", {"val": "autoZero"})
    _chart_node(category, "auto", {"val": 1})
    _chart_node(category, "lblAlgn", {"val": "ctr"})
    _chart_node(category, "lblOffset", {"val": 100})
    value = _chart_node(plot, "valAx")
    _chart_node(value, "axId", {"val": 200})
    _chart_node(_chart_node(value, "scaling"), "orientation", {"val": "minMax"})
    _chart_node(value, "delete", {"val": 0})
    _chart_node(value, "axPos", {"val": "l"})
    grid = _chart_node(value, "majorGridlines")
    _art_node(_art_node(_art_node(_chart_node(grid, "spPr"), "ln"), "solidFill"),
              "srgbClr", {"val": "DFE7EE"})
    if spec["y_axis_title"]:
        _chart_title(value, spec["y_axis_title"], size=1000)
    _chart_node(value, "numFmt", {"formatCode": "0.000", "sourceLinked": 0})
    _chart_node(value, "majorTickMark", {"val": "none"})
    _chart_node(value, "minorTickMark", {"val": "none"})
    _chart_node(value, "tickLblPos", {"val": "nextTo"})
    _chart_node(value, "crossAx", {"val": 100})
    _chart_node(value, "crosses", {"val": "autoZero"})
    _chart_node(value, "crossBetween", {"val": "between"})
    legend = _chart_node(chart, "legend")
    _chart_node(legend, "legendPos", {"val": "b"})
    _chart_node(legend, "layout")
    _chart_node(legend, "overlay", {"val": 0})
    _chart_node(chart, "plotVisOnly", {"val": 1})
    _chart_node(chart, "dispBlanksAs", {"val": "gap"})
    _chart_node(chart, "showDLblsOverMax", {"val": 0})
    _art_node(_art_node(_chart_node(root, "spPr"), "solidFill"), "srgbClr", {"val": "FFFFFF"})
    return root


def _chart_drawing_xml(charts):
    root = ET.Element("{" + DRAWING + "}wsDr")
    for index, spec in enumerate(charts, 1):
        node = _drawing_node(root, DRAWING, "twoCellAnchor", {"editAs": "oneCell"})
        r1, c1, r2, c2 = spec["anchor"]
        for name, row, col in (("from", r1, c1), ("to", r2, c2)):
            corner = _drawing_node(node, DRAWING, name)
            for key, value in (("col", col - 1), ("colOff", 0), ("row", row - 1), ("rowOff", 0)):
                _drawing_node(corner, DRAWING, key, text=value)
        frame = _drawing_node(node, DRAWING, "graphicFrame", {"macro": ""})
        props = _drawing_node(frame, DRAWING, "nvGraphicFramePr")
        _drawing_node(props, DRAWING, "cNvPr", {"id": index, "name": "Chart " + str(index)})
        _drawing_node(props, DRAWING, "cNvGraphicFramePr")
        transform = _drawing_node(frame, DRAWING, "xfrm")
        _art_node(transform, "off", {"x": 0, "y": 0})
        _art_node(transform, "ext", {"cx": 0, "cy": 0})
        graphic = _art_node(frame, "graphic")
        data = _art_node(graphic, "graphicData", {"uri": CHART})
        _chart_node(data, "chart", {"{" + REL + "}id": "rId" + str(index)})
        _drawing_node(node, DRAWING, "clientData")
    return root


def _text(value):
    value = str(value)
    if any(not (char in "\t\n\r" or 0x20 <= ord(char) <= 0xD7FF
                or 0xE000 <= ord(char) <= 0xFFFD
                or 0x10000 <= ord(char) <= 0x10FFFF) for char in value):
        raise ValueError("Excel text contains an invalid XML character")
    return value


def _position(row, col):
    if (type(row) is not int or type(col) is not int
            or not 1 <= row <= MAX_ROW or not 1 <= col <= MAX_COL):
        raise ValueError("Excel row/column is outside the one-based worksheet range: "
                         f"row={row!r} ({type(row).__name__}), column={col!r} ({type(col).__name__}); "
                         f"allowed rows=1..{MAX_ROW}, columns=1..{MAX_COL}")


def column_name(col):
    """Convert a one-based column number to its Excel letters."""
    _position(1, col)
    result = ""
    while col:
        col, remainder = divmod(col - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_reference(row, col, absolute=False):
    _position(row, col)
    return ("$" if absolute else "") + column_name(col) + ("$" if absolute else "") + str(row)


def _value_type(value):
    if value is None:
        return "blank", None
    if isinstance(value, bool):
        return "b", "1" if value else "0"
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            raise ValueError("Excel numeric values must be finite")
        return "n", str(value)
    return "str", _text(value)


# The first font is the ordinary workbook font.  Every style includes an
# explicit font, fill, border and alignment; no installed Office theme is used.
STYLE_DEFINITIONS = {
    "title": (1, 2, 0, "left", 0),
    "subtitle": (2, 2, 0, "left", 0),
    "section": (3, 3, 0, "left", 1),
    "header": (3, 2, 0, "center", 1),
    "body": (0, 0, 0, "left", 1),
    "label": (4, 4, 0, "left", 1),
    "number": (0, 0, 164, "right", 1),
    "integer": (0, 0, 165, "right", 1),
    "input": (4, 5, 164, "right", 1),
    "formula": (4, 6, 164, "right", 1),
    "formula_input": (4, 5, 164, "right", 1),
    "note": (5, 0, 0, "left", 0),
    "warning": (6, 5, 0, "left", 1),
    "total": (4, 7, 164, "right", 1),
    "hero": (7, 3, 167, "right", 1),
    "hero_label": (8, 3, 0, "left", 1),
    "percent": (0, 0, 166, "right", 1),
    "unit": (5, 4, 0, "center", 1),
    "display": (0, 0, 167, "right", 1),
    "reference": (0, 0, 168, "right", 1),
    "ratio": (4, 6, 169, "right", 1),
    "total_display": (4, 7, 167, "right", 1),
    "hero_integer": (7, 3, 165, "right", 1),
}
STYLE_IDS = {name: index for index, name in enumerate(STYLE_DEFINITIONS, 1)}


def _styles():
    root = ET.Element("{" + MAIN + "}styleSheet")
    formats = _node(root, "numFmts", {"count": 6})
    for number, code in ((164, "0.000000"), (165, "#,##0"), (166, "0.0%"),
                         (167, "#,##0.000"), (168, "0.00"), (169, '0.000"×"')):
        _node(formats, "numFmt", {"numFmtId": number, "formatCode": code})
    fonts = _node(root, "fonts", {"count": 9})
    for size, color, bold in ((10, "24364B", False), (23, "FFFFFF", True),
                              (11, "E3EDF7", False), (11, "FFFFFF", True),
                              (10, "16314F", True), (9, "617286", False),
                              (10, "9A4B10", True), (20, "FFFFFF", True),
                              (12, "FFFFFF", True)):
        font = _node(fonts, "font")
        _node(font, "sz", {"val": size})
        _node(font, "color", {"rgb": "FF" + color})
        _node(font, "name", {"val": "微软雅黑"})
        _node(font, "family", {"val": 2})
        _node(font, "charset", {"val": 134})
        if bold:
            _node(font, "b")
    fills = _node(root, "fills", {"count": 8})
    for pattern, color in (("none", None), ("gray125", None),
                            ("solid", "173451"), ("solid", "127D87"),
                            ("solid", "EDF2F7"), ("solid", "FFF0DB"),
                            ("solid", "E5F5F1"), ("solid", "DCECF4")):
        fill = _node(fills, "fill")
        pattern_fill = _node(fill, "patternFill", {"patternType": pattern})
        if color:
            _node(pattern_fill, "fgColor", {"rgb": "FF" + color})
            _node(pattern_fill, "bgColor", {"indexed": 64})
    borders = _node(root, "borders", {"count": 2})
    for lined in (False, True):
        border = _node(borders, "border")
        for edge in ("left", "right", "top", "bottom", "diagonal"):
            child = _node(border, edge, {"style": "thin"} if lined and edge != "diagonal" else {})
            if lined and edge != "diagonal":
                _node(child, "color", {"rgb": "FFD2DDE8"})
    bases = _node(root, "cellStyleXfs", {"count": 1})
    _node(bases, "xf", {"numFmtId": 0, "fontId": 0, "fillId": 0, "borderId": 0})
    cell_xfs = _node(root, "cellXfs", {"count": len(STYLE_DEFINITIONS) + 1})
    _node(cell_xfs, "xf", {"numFmtId": 0, "fontId": 0, "fillId": 0, "borderId": 0, "xfId": 0})
    for font, fill, number, horizontal, border in STYLE_DEFINITIONS.values():
        xf = _node(cell_xfs, "xf", {"numFmtId": number, "fontId": font,
                                    "fillId": fill, "borderId": border, "xfId": 0,
                                    "applyAlignment": 1, "applyFont": 1,
                                    "applyFill": 1, "applyBorder": 1,
                                    "applyNumberFormat": 1})
        _node(xf, "alignment", {"horizontal": horizontal, "vertical": "center", "wrapText": 1})
    cell_styles = _node(root, "cellStyles", {"count": 1})
    _node(cell_styles, "cellStyle", {"name": "Normal", "xfId": 0, "builtinId": 0})
    return root


class Worksheet:
    def __init__(self, name, widths, landscape, paper_size, print_area):
        self.name = name
        self.widths = list(widths or [])
        if len(self.widths) > MAX_COL or any(not isinstance(width, numbers.Real)
                or not math.isfinite(width) or width <= 0 or width > 255 for width in self.widths):
            raise ValueError("Column widths must be finite numbers between 0 and 255")
        if type(paper_size) is not int or not 1 <= paper_size <= 118:
            raise ValueError("paper_size must be a valid Excel paper-size number (A3=8, A4=9)")
        self.landscape = bool(landscape)
        self.paper_size = paper_size
        self.print_area = print_area
        self.cells = {}
        self.merges = []
        self.heights = {}
        self.frozen = (0, 0)
        self.charts = []

    def _check_position(self, row, col):
        try:
            _position(row, col)
        except ValueError as error:
            raise ValueError(f"Worksheet={self.name!r}; {error}") from error

    def _put(self, row, col, value, style, expression=None):
        self._check_position(row, col)
        if style not in STYLE_IDS:
            raise ValueError("Unknown worksheet style: " + str(style))
        for r1, c1, r2, c2 in self.merges:
            if r1 <= row <= r2 and c1 <= col <= c2 and (row, col) != (r1, c1):
                raise ValueError("Cannot write into a non-anchor merged cell")
        value_type, serialized = _value_type(value)
        self.cells[row, col] = (value_type, serialized, STYLE_IDS[style], expression)
        return self

    def write(self, row, col, value, style="body"):
        return self._put(row, col, value, style)

    def formula(self, row, col, expression, cached_value, style="number"):
        expression = _text(expression)
        if expression.startswith("="):
            expression = expression[1:]
        if not expression.strip():
            raise ValueError("A worksheet formula must not be empty")
        return self._put(row, col, cached_value, style, expression)

    def merge(self, r1, c1, r2, c2, value, style="title"):
        self._check_position(r1, c1)
        self._check_position(r2, c2)
        if r1 > r2 or c1 > c2:
            raise ValueError("Merged range is reversed")
        if any(not (r2 < a or r1 > b or c2 < c or c1 > d)
               for a, c, b, d in self.merges):
            raise ValueError("Merged ranges must not overlap")
        if any(r1 <= row <= r2 and c1 <= col <= c2 and (row, col) != (r1, c1)
               for row, col in self.cells):
            raise ValueError("Merged range would hide existing cells")
        self.write(r1, c1, value, style)
        if (r1, c1) != (r2, c2):
            self.merges.append((r1, c1, r2, c2))
        return self

    def row_height(self, row, pt):
        self._check_position(row, 1)
        if not isinstance(pt, numbers.Real) or not math.isfinite(pt) or not 0 < pt <= 409:
            raise ValueError("Row height must be a finite value between 0 and 409 points")
        self.heights[row] = pt
        return self

    def freeze(self, row, col):
        if (type(row) is not int or type(col) is not int
                or not 0 <= row < MAX_ROW or not 0 <= col < MAX_COL):
            raise ValueError("Freeze counts must be nonnegative row/column counts")
        self.frozen = row, col
        return self

    def add_bar_chart(self, title, categories, series, anchor, stacked=True,
                      x_axis_title="耗时（ms）", categories_ref=None):
        """Add a native, editable horizontal Excel bar chart.

        ``anchor=(r1,c1,r2,c2)`` uses one-based worksheet corners; the chart
        extends from the top-left of the first cell to the top-left of the
        second.  Each series is a mapping with ``name``, finite numeric
        ``values``, optional six-digit RGB ``color``, and optional Excel
        ``values_ref`` and ``name_ref`` formulas (without the leading '=').
        ``categories_ref`` is an optional range formula for all series.
        References always include supplied caches, so the same chart works in
        readers before recalculation and updates when cells change in Excel.
        Unknown values must be kept out of charts; no missing value becomes 0.
        """
        return self._add_chart("bar", title, categories, series, anchor, stacked=stacked,
                               x_axis_title=x_axis_title, categories_ref=categories_ref)

    def add_line_chart(self, title, categories, series, anchor, x_axis_title="BS",
                       y_axis_title="", categories_ref=None):
        """Add an editable line chart in the supplied category order.

        Series accept the same name, color and range references as bar charts.
        ``None`` leaves a missing point at its original index; Excel displays
        a gap and never substitutes zero. Finite zeros remain measured points.
        """
        return self._add_chart("line", title, categories, series, anchor,
                               x_axis_title=x_axis_title, y_axis_title=y_axis_title,
                               categories_ref=categories_ref)

    def _add_chart(self, kind, title, categories, series, anchor, *, stacked=False,
                   x_axis_title="", y_axis_title="", categories_ref=None):
        title, x_axis_title = _text(title), _text(x_axis_title)
        if not title.strip():
            raise ValueError("A chart title must not be empty")
        if isinstance(categories, (str, bytes)):
            raise ValueError("Chart categories must be a sequence of labels")
        categories = [_text(value) for value in categories]
        if not categories or any(not value.strip() for value in categories):
            raise ValueError("Chart categories must contain nonempty labels")
        if len(anchor) != 4:
            raise ValueError("Chart anchor requires four one-based coordinates")
        r1, c1, r2, c2 = anchor
        self._check_position(r1, c1)
        self._check_position(r2, c2)
        if r1 >= r2 or c1 >= c2:
            raise ValueError("Chart anchor must have a positive height and width")

        def reference(value):
            if value is None:
                return None
            value = _text(value).strip().removeprefix("=")
            if not value or any(char in value for char in "\r\n"):
                raise ValueError("Chart range references must be nonempty Excel formulas")
            return value

        colors = ("167D8D", "4F90C6", "D59440", "7668AA", "5BA28A", "D06F6F", "6C8498")
        normalized = []
        for index, item in enumerate(series):
            if not isinstance(item, dict):
                raise ValueError("Chart series must be dictionaries")
            name = _text(item.get("name", ""))
            if not name.strip():
                raise ValueError("Every chart series requires a nonempty name")
            values = list(item.get("values", []))
            if len(values) != len(categories):
                raise ValueError("Chart series and categories must have the same length")
            if any((value is None and kind != "line") or value is not None and
                   (isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value))
                   for value in values):
                raise ValueError("Chart values must be finite numbers, not unknown or Boolean values"
                                 if kind == "bar" else "Line chart values must be finite numbers or None")
            color = str(item.get("color", colors[index % len(colors)])).upper()
            if not re.fullmatch(r"[0-9A-F]{6}", color):
                raise ValueError("Chart colors must be six hexadecimal RGB digits")
            normalized.append({"name": name, "values": values, "color": color,
                               "values_ref": reference(item.get("values_ref")),
                               "name_ref": reference(item.get("name_ref"))})
        if not normalized:
            raise ValueError("A " + kind + " chart requires at least one series")
        spec = {"title": title, "categories": categories,
                "series": normalized, "anchor": tuple(anchor),
                "stacked": bool(stacked), "x_axis_title": x_axis_title,
                "categories_ref": reference(categories_ref)}
        if kind == "line":
            spec.update(kind="line", y_axis_title=_text(y_axis_title))
        self.charts.append(spec)
        return self

    def _extent(self):
        return (max([1, *self.heights, *(row for row, _ in self.cells), *(m[2] for m in self.merges),
                     *(chart["anchor"][2] for chart in self.charts)]),
                max([1, *(col for _, col in self.cells), *(m[3] for m in self.merges),
                     *(chart["anchor"][3] for chart in self.charts)]))

    def _print_range(self):
        if self.print_area is None:
            values = (1, 1, *self._extent())
        elif isinstance(self.print_area, str):
            match = re.fullmatch(r"\$?([A-Za-z]+)\$?(\d+)(?::\$?([A-Za-z]+)\$?(\d+))?", self.print_area.strip())
            if not match:
                raise ValueError("print_area must be A1:H50 or a four-number (r1,c1,r2,c2) tuple")
            a, b, c, d = match.groups()
            def col_number(letters):
                result = 0
                for letter in letters.upper():
                    result = result * 26 + ord(letter) - 64
                return result
            values = int(b), col_number(a), int(d or b), col_number(c or a)
        else:
            values = tuple(self.print_area)
            if len(values) != 4:
                raise ValueError("print_area requires four one-based coordinates")
        r1, c1, r2, c2 = values
        self._check_position(r1, c1)
        self._check_position(r2, c2)
        if r1 > r2 or c1 > c2:
            raise ValueError("Print range is reversed")
        sheet_name = self.name.replace("'", "''")
        return "'" + sheet_name + "'!" + cell_reference(r1, c1, True) + ":" + cell_reference(r2, c2, True)

    def _xml(self):
        root = ET.Element("{" + MAIN + "}worksheet")
        properties = _node(root, "sheetPr")
        _node(properties, "pageSetUpPr", {"fitToPage": 1})
        row_max, col_max = self._extent()
        _node(root, "dimension", {"ref": "A1:" + cell_reference(row_max, col_max)})
        views = _node(root, "sheetViews")
        view = _node(views, "sheetView", {"showGridLines": 0, "workbookViewId": 0})
        frozen_rows, frozen_cols = self.frozen
        if frozen_rows or frozen_cols:
            pane_name = "bottomRight" if frozen_rows and frozen_cols else "bottomLeft" if frozen_rows else "topRight"
            attributes = {"topLeftCell": cell_reference(frozen_rows + 1, frozen_cols + 1),
                          "activePane": pane_name, "state": "frozen"}
            if frozen_rows:
                attributes["ySplit"] = frozen_rows
            if frozen_cols:
                attributes["xSplit"] = frozen_cols
            _node(view, "pane", attributes)
            _node(view, "selection", {"pane": pane_name,
                                      "activeCell": attributes["topLeftCell"], "sqref": attributes["topLeftCell"]})
        _node(root, "sheetFormatPr", {"defaultRowHeight": 20})
        if self.widths:
            columns = _node(root, "cols")
            for index, width in enumerate(self.widths, 1):
                _node(columns, "col", {"min": index, "max": index, "width": width, "customWidth": 1})
        sheet_data = _node(root, "sheetData")
        by_row = {}
        for (row, col), value in self.cells.items():
            by_row.setdefault(row, []).append((col, value))
        for row in sorted(set(by_row) | set(self.heights)):
            attributes = {"r": row}
            if row in self.heights:
                attributes.update(ht=self.heights[row], customHeight=1)
            row_node = _node(sheet_data, "row", attributes)
            for col, (kind, serialized, style, expression) in sorted(by_row.get(row, [])):
                attrs = {"r": cell_reference(row, col), "s": style}
                if kind == "str":
                    attrs["t"] = "str" if expression is not None else "inlineStr"
                elif kind == "b":
                    attrs["t"] = "b"
                cell = _node(row_node, "c", attrs)
                if expression is not None:
                    _node(cell, "f", text=expression)
                    if kind != "blank":
                        _node(cell, "v", text=serialized)
                elif kind == "str":
                    inline = _node(cell, "is")
                    _node(inline, "t", {"{" + XML + "}space": "preserve"}, serialized)
                elif kind != "blank":
                    _node(cell, "v", text=serialized)
        if self.merges:
            merged = _node(root, "mergeCells", {"count": len(self.merges)})
            for r1, c1, r2, c2 in self.merges:
                _node(merged, "mergeCell", {"ref": cell_reference(r1, c1) + ":" + cell_reference(r2, c2)})
        _node(root, "printOptions", {"horizontalCentered": 1})
        _node(root, "pageMargins", {"left": .25, "right": .25, "top": .35, "bottom": .35,
                                    "header": .15, "footer": .15})
        _node(root, "pageSetup", {"paperSize": self.paper_size,
                                  "orientation": "landscape" if self.landscape else "portrait",
                                  "fitToWidth": 1, "fitToHeight": 0})
        footer = _node(root, "headerFooter")
        _node(footer, "oddFooter", text="&L" + self.name + "&C第 &P / &N 页")
        if self.charts:
            _node(root, "drawing", {"{" + REL + "}id": "rIdDrawing"})
        return root


class Workbook:
    def __init__(self):
        self.sheets = []

    def add_sheet(self, name, widths=None, landscape=True, paper_size=8, print_area=None):
        if (not isinstance(name, str) or not 1 <= len(name) <= 31
                or re.search(r"[\[\]:*?/\\]", name) or name.startswith("'") or name.endswith("'")):
            raise ValueError("Worksheet names require 1–31 characters without []:*?/\\ or edge apostrophes")
        _text(name)
        if any(sheet.name.casefold() == name.casefold() for sheet in self.sheets):
            raise ValueError("Worksheet names must be unique")
        sheet = Worksheet(name, widths, landscape, paper_size, print_area)
        self.sheets.append(sheet)
        return sheet

    def save(self, path):
        if not self.sheets:
            raise ValueError("An XLSX workbook must contain at least one worksheet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook = ET.Element("{" + MAIN + "}workbook")
        sheets = _node(workbook, "sheets")
        names = _node(workbook, "definedNames")
        relationships = ET.Element("Relationships", {"xmlns": PACKAGE_REL})
        content = ET.Element("Types", {"xmlns": CONTENT})
        ET.SubElement(content, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
        ET.SubElement(content, "Default", Extension="xml", ContentType="application/xml")
        overrides = [("/xl/workbook.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"),
                     ("/xl/styles.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"),
                     ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
                     ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml")]
        chart_parts = []
        chart_number = 0
        for index, sheet in enumerate(self.sheets, 1):
            _node(sheets, "sheet", {"name": sheet.name, "sheetId": index, "{" + REL + "}id": "rId" + str(index)})
            _node(names, "definedName", {"name": "_xlnm.Print_Area", "localSheetId": index - 1}, sheet._print_range())
            ET.SubElement(relationships, "Relationship", Id="rId" + str(index), Type=REL + "/worksheet", Target="worksheets/sheet" + str(index) + ".xml")
            overrides.append(("/xl/worksheets/sheet" + str(index) + ".xml",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"))
            if sheet.charts:
                drawing_name = "drawing" + str(index) + ".xml"
                sheet_rels = ET.Element("Relationships", {"xmlns": PACKAGE_REL})
                ET.SubElement(sheet_rels, "Relationship", Id="rIdDrawing", Type=REL + "/drawing",
                              Target="../drawings/" + drawing_name)
                chart_parts.append(("xl/worksheets/_rels/sheet" + str(index) + ".xml.rels", sheet_rels))
                chart_parts.append(("xl/drawings/" + drawing_name, _chart_drawing_xml(sheet.charts)))
                overrides.append(("/xl/drawings/" + drawing_name,
                                  "application/vnd.openxmlformats-officedocument.drawing+xml"))
                drawing_rels = ET.Element("Relationships", {"xmlns": PACKAGE_REL})
                for local_index, spec in enumerate(sheet.charts, 1):
                    chart_number += 1
                    chart_name = "chart" + str(chart_number) + ".xml"
                    chart_xml = _line_chart_xml(spec) if spec.get("kind") == "line" else _bar_chart_xml(spec)
                    chart_parts.append(("xl/charts/" + chart_name, chart_xml))
                    overrides.append(("/xl/charts/" + chart_name,
                                      "application/vnd.openxmlformats-officedocument.drawingml.chart+xml"))
                    ET.SubElement(drawing_rels, "Relationship", Id="rId" + str(local_index),
                                  Type=REL + "/chart", Target="../charts/" + chart_name)
                chart_parts.append(("xl/drawings/_rels/" + drawing_name + ".rels", drawing_rels))
        _node(workbook, "calcPr", {"calcId": 191029, "calcMode": "auto", "fullCalcOnLoad": 1, "forceFullCalc": 1})
        ET.SubElement(relationships, "Relationship", Id="rId" + str(len(self.sheets) + 1), Type=REL + "/styles", Target="styles.xml")
        for part, content_type in overrides:
            ET.SubElement(content, "Override", PartName=part, ContentType=content_type)
        package = ET.Element("Relationships", {"xmlns": PACKAGE_REL})
        for number, kind, target in ((1, REL + "/officeDocument", "xl/workbook.xml"),
                                     (2, "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
                                     (3, REL + "/extended-properties", "docProps/app.xml")):
            ET.SubElement(package, "Relationship", Id="rId" + str(number), Type=kind, Target=target)
        core_ns = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
        dc = "http://purl.org/dc/elements/1.1/"
        dcterms = "http://purl.org/dc/terms/"
        # Bind the QName used by xsi:type through the actual created element.
        # A literal xmlns attribute would duplicate a namespace registered by
        # another XML consumer (for example openpyxl) in the same process.
        ET.register_namespace("dcterms", dcterms)
        core = ET.Element("{" + core_ns + "}coreProperties")
        ET.SubElement(core, "{" + dc + "}creator").text = "EP128 profiling report"
        ET.SubElement(core, "{" + dcterms + "}created", {
            "{http://www.w3.org/2001/XMLSchema-instance}type": "dcterms:W3CDTF"}).text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        application = ET.Element("Properties", {"xmlns": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"})
        ET.SubElement(application, "Application").text = "EP128 profiling report"
        descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
                for filename, root in (("[Content_Types].xml", content), ("_rels/.rels", package),
                                       ("docProps/core.xml", core), ("docProps/app.xml", application),
                                       ("xl/workbook.xml", workbook), ("xl/_rels/workbook.xml.rels", relationships),
                                       ("xl/styles.xml", _styles())):
                    archive.writestr(filename, _xml(root))
                for index, sheet in enumerate(self.sheets, 1):
                    archive.writestr("xl/worksheets/sheet" + str(index) + ".xml", _xml(sheet._xml()))
                for filename, root in chart_parts:
                    archive.writestr(filename, _xml(root))
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path
