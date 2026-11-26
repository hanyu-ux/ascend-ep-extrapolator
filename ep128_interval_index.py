"""Exact static containment queries that keep long enclosing trace scopes.

Inputs are already start-sorted. The index returns their original positions,
in ascending order, and never shortens an interval or selects one owner. A
maximum-end tree skips only ranges whose every interval fails the same endpoint
predicate as the direct scan. Broad Python scopes therefore do not force each
query to rescan all earlier, already finished children.
"""

from bisect import bisect_left, bisect_right


class MaxEndIntervalIndex:
    """Index immutable snapshots of parallel start/end integer sequences."""

    def __init__(self, starts, ends):
        self._starts, self._ends = tuple(starts), tuple(ends)
        if len(self._starts) != len(self._ends):
            raise ValueError("Interval starts and ends must have the same length")
        if any(left > right for left, right in zip(self._starts, self._starts[1:])):
            raise ValueError("Interval starts must be sorted")
        maximum, prefix = None, []
        for end in self._ends:
            maximum = end if maximum is None else max(maximum, end)
            prefix.append(maximum)
        self._prefix = tuple(prefix)
        size = 1
        while size < len(self._ends):
            size *= 2
        self._size = size
        # None rather than an integer sentinel preserves every exact timestamp,
        # including negative times and values beyond signed 64-bit range.
        tree = [None] * (2*size)
        tree[size:size+len(self._ends)] = self._ends
        for node in range(size-1, 0, -1):
            left, right = tree[2*node], tree[2*node+1]
            tree[node] = right if left is None else left if right is None else max(left, right)
        self._max_ends = tree

    def containing(self, start, end=None, *, include_end=False):
        """Return positions satisfying ``left <= start < right``.

        When ``end`` is supplied, also require ``right >= end``. The optional
        ``include_end`` makes only the point predicate right-inclusive for raw
        boundary-event retention; callers choose whether that is appropriate.
        It does not establish a submission or Python ownership relationship.
        """
        stop = bisect_right(self._starts, start)
        first = (bisect_left if include_end else bisect_right)(self._prefix, start)
        if stop <= first:
            return []
        # The old prefix query is efficient for short/nonoverlapping scopes.
        # Preserve that fast path, using the tree for broad/nested lifetimes.
        if stop-first <= 16:
            return [position for position in range(first, stop)
                    if (self._ends[position] >= start if include_end else self._ends[position] > start)
                    and (end is None or self._ends[position] >= end)]
        result, pending = [], [(1, 0, self._size)]
        while pending:
            node, left, right = pending.pop()
            maximum = self._max_ends[node]
            if (left >= stop or right <= first or maximum is None
                    or (maximum < start if include_end else maximum <= start)
                    or end is not None and maximum < end):
                continue
            if right-left == 1:
                result.append(left)
                continue
            middle = (left+right)//2
            # Visit left first so positions retain the caller's exact order.
            pending.append((2*node+1, middle, right))
            pending.append((2*node, left, middle))
        return result
