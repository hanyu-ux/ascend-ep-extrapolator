"""Exact interval indexing preserves source CSVs and avoids broad-scope scans."""

import ast
from bisect import bisect_right
from copy import deepcopy
import inspect
from pathlib import Path
import random
import sqlite3
import tempfile
import textwrap
from types import SimpleNamespace
import unittest

from ep128_interval_index import MaxEndIntervalIndex
import profile_ep128 as core


def linear_positions(starts, ends, start, end=None, include_end=False):
    return [position for position in range(bisect_right(starts, start))
            if (ends[position] >= start if include_end else ends[position] > start)
            and (end is None or ends[position] >= end)]


def linear_source_rows():
    """Use the direct containment predicate as an independent query oracle."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(core.residual_source_rows)))
    replacement = ast.parse('''
def containing(kind, pid, tid, start, end=None):
    vectors = intervals.get((kind, pid if kind == "cpu" else None, tid))
    if vectors is None:
        return []
    starts, ends, identities, _ = vectors
    return [event(identities[position]) for position in range(bisect.bisect_right(starts, start))
            if ends[position] > start and (end is None or ends[position] >= end)]
''').body[0]
    tree.body[0].body = [replacement if isinstance(node, ast.FunctionDef) and node.name == "containing" else node
                         for node in tree.body[0].body]
    ast.fix_missing_locations(tree)
    namespace = dict(vars(core))
    exec(compile(tree, "<test-linear-source-intervals>", "exec"), namespace)
    return namespace["residual_source_rows"]


def source_database(count=40, mode="unique"):
    connection = sqlite3.connect(":memory:")
    connection.executescript('''
CREATE TABLE source_events(event_index INTEGER PRIMARY KEY,kind TEXT,pid TEXT,tid TEXT,name TEXT,
start INTEGER,end INTEGER,connection_id TEXT,correlation_id TEXT,category TEXT);
CREATE TABLE source_flows(cat TEXT,flow_key TEXT,ph TEXT,pid TEXT,tid TEXT,stamp INTEGER);
CREATE TABLE source_devices(row_index INTEGER,pid TEXT,tid TEXT,start INTEGER,end INTEGER,connection_id TEXT);
''')
    epoch = 1790000000000000003
    events = [(1, "cpu", "100", "101", "whole profiling scope", epoch, epoch+count*1000+2000,
               None, None, "python_function")]
    flows, devices, tasks, stages = [], [], [], []
    for number in range(count):
        point, identity = epoch+1000*(number+1), 10+number*8
        native_connection = str(number % 3 if mode == "reused_connection" else number)
        correlation = str(number % 2 if mode == "reused_correlation" else number)
        events.extend([
            (identity, "cpu", "100", "101", "outer Python", point, point+60, None, None, "python_function"),
            (identity+1, "cpu", "100", "101", "inner Python", point+1, point+59, None, None, "python_function"),
            (identity+2, "cpu", "100", "101", "aten::metadata", point+2, point+58, None, None, "cpu_op"),
            (identity+3, "enqueue", "100", "101", "Enqueue@metadata", point+10, point+11, None, correlation, "enqueue"),
            (identity+4, "dequeue", "100", "102", "Dequeue@metadata", point+20, point+30, None, correlation, "dequeue"),
            (identity+5, "node", "200", "102", "Node@metadata", point+21, point+25, native_connection, None, "cann_api"),
        ])
        flow_key = "flow-"+str(number)
        flows.extend([("HostToDevice", flow_key, "s", "200", "102", point+22),
                      ("HostToDevice", flow_key, "f", "300", "301", point+100)])
        devices.append((number+1, "300", "301", point+100, point+120, native_connection))
        tasks.append(SimpleNamespace(row_index=number+1, name="Metadata", start_ns=point+100, end_ns=point+120))
        stages.append({"round_id":str(number+1), "window_key":"source_only",
                       "start_ns":str(point+100), "end_ns":str(point+120), "_task_rows":[number+1]})
    connection.executemany("INSERT INTO source_events VALUES (?,?,?,?,?,?,?,?,?,?)", events)
    connection.executemany("INSERT INTO source_flows VALUES (?,?,?,?,?,?)", flows)
    connection.executemany("INSERT INTO source_devices VALUES (?,?,?,?,?,?)", devices)
    connection.commit()
    return connection, tasks, stages


class IntervalIndexTests(unittest.TestCase):
    def test_exact_boundaries_order_empty_and_large_integer_times(self):
        epoch = 2**100
        starts = [epoch-10, epoch, epoch, epoch+5]
        ends = [epoch+100, epoch+5, epoch+8, epoch+10]
        index = MaxEndIntervalIndex(starts, ends)
        self.assertEqual(index.containing(epoch+5), [0, 2, 3])
        self.assertEqual(index.containing(epoch+5, include_end=True), [0, 1, 2, 3])
        self.assertEqual(index.containing(epoch+5, epoch+10), [0, 3])
        self.assertEqual(index.containing(epoch+5, epoch+10, include_end=True), [0, 3])
        self.assertEqual(index.containing(epoch-11), [])
        self.assertEqual(MaxEndIntervalIndex([], []).containing(epoch), [])

    def test_static_snapshot_and_invalid_inputs(self):
        starts, ends = [0, 2], [10, 3]
        index = MaxEndIntervalIndex(starts, ends)
        starts[0], ends[0] = 100, -1
        self.assertEqual(index.containing(2), [0, 1])
        with self.assertRaises(ValueError):
            MaxEndIntervalIndex([2, 1], [3, 4])
        with self.assertRaises(ValueError):
            MaxEndIntervalIndex([1], [])

    def test_random_nested_disjoint_duplicate_and_zero_intervals_match_scan(self):
        generator = random.Random(128)
        for _ in range(40):
            starts = sorted(generator.randrange(-40, 150) for _ in range(120))
            ends = [start+generator.randrange(0, 250) for start in starts]
            starts.insert(0, -100)
            ends.insert(0, 1000)  # an enclosing scope must not be removed
            index = MaxEndIntervalIndex(starts, ends)
            for _ in range(30):
                point = generator.randrange(-100, 1000)
                end = None if generator.randrange(2) else point+generator.randrange(0, 300)
                for inclusive in (False, True):
                    self.assertEqual(index.containing(point, end, include_end=inclusive),
                                     linear_positions(starts, ends, point, end, inclusive))

    def test_long_scope_does_not_scan_all_prior_finished_children(self):
        count = 20000
        starts = [0]+[10*number for number in range(1, count+1)]
        ends = [10*count+100]+[start+3 for start in starts[1:]]
        index = MaxEndIntervalIndex(starts, ends)

        class CountingList(list):
            reads = 0

            def __getitem__(self, position):
                self.reads += 1
                return super().__getitem__(position)

        index._max_ends = CountingList(index._max_ends)
        index._ends = CountingList(index._ends)
        self.assertEqual(index.containing(10*count+1), [0, count])
        # Deterministic operation bound, not a machine-dependent wall-time test.
        self.assertLess(index._max_ends.reads+index._ends.reads, 160)


class SourceIntervalDifferentialTests(unittest.TestCase):
    def test_all_source_csv_bytes_and_stage_summaries_match_direct_scan(self):
        oracle = linear_source_rows()
        for mode in ("unique", "reused_connection", "reused_correlation"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output, stages, summaries = [], [], []
                for name, function in (("linear", oracle), ("indexed", core.residual_source_rows)):
                    connection, tasks, rows = source_database(mode=mode)
                    try:
                        destination = Path(directory)/(name+".csv")
                        summary = function(connection, tasks, rows, destination, None)
                        output.append(destination.read_bytes())
                        stages.append(deepcopy(rows))
                        summaries.append({key:value for key,value in summary.items() if key != "source_links_csv"})
                    finally:
                        connection.close()
                self.assertEqual(output[0], output[1])
                self.assertEqual(stages[0], stages[1])
                self.assertEqual(summaries[0], summaries[1])
                self.assertEqual(summaries[1]["source_status"], "UNAVAILABLE" if mode == "reused_correlation" else "COMPLETE")


if __name__ == "__main__":
    unittest.main()
