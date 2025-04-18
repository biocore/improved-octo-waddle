import math

class BitVector:
    def __init__(self, bitlist):
        self.B = bitlist
        self.n = len(bitlist)
        self.block_size = max(1, int(math.log2(self.n)))
        self.superblock_size = self.block_size * self.block_size

        # Precompute rank directories
        self.superblocks = []
        self.blocks = []
        self.precompute_ranks()

    def precompute_ranks(self):
        rank = 0
        for i in range(0, self.n, self.superblock_size):
            self.superblocks.append(rank)
            for j in range(i, min(i + self.superblock_size, self.n), self.block_size):
                self.blocks.append(rank)
                rank += sum(self.B[j:j + self.block_size])

    def rank1(self, idx):
        """Return the number of 1s up to and including index idx."""
        if idx < 0 or idx >= self.n:
            raise IndexError("Index out of bounds")
        superblock_idx = idx // self.superblock_size
        block_idx = idx // self.block_size
        rank = self.superblocks[superblock_idx]
        rank += sum(self.B[superblock_idx * self.superblock_size:(block_idx * self.block_size)])
        rank += sum(self.B[block_idx * self.block_size:idx + 1])
        return rank

    def rank0(self, idx):
        """Return the number of 0s up to and including index idx."""
        return idx + 1 - self.rank1(idx)

class BP_Tree:
    def __init__(self, bitvector):
        self.bitvector = bitvector

class RMMinMaxTree:
    def __init__(self, bp_tree):
        self.bp_tree = bp_tree
        self.bitvector = bp_tree.bitvector
        self.n = self.bitvector.n
        self.initialize_blocks()
        self.build_blocks()
        self.precompute_miniblock_tables()

    def initialize_blocks(self):
        n = self.n
        # Superblock size: β = 0.5 * log n
        self.superblock_size = max(1, int(0.5 * math.log2(n)))
        # Miniblock size: γ = 0.5 * log log n
        self.miniblock_size = max(1, int(0.5 * math.log2(max(2, math.log2(n)))))
        self.num_superblocks = (n + self.superblock_size - 1) // self.superblock_size
        self.superblocks = []

    def build_blocks(self):
        # Build superblocks
        current_excess = 0
        for b in range(self.num_superblocks):
            start = b * self.superblock_size
            end = min((b + 1) * self.superblock_size, self.n)
            superblock = {
                'start': start,
                'end': end - 1,
                'e': 0,
                'm': float('inf'),
                'M': -float('inf'),
                'miniblocks': []
            }
            sb_initial_excess = current_excess
            sb_min_excess = float('inf')
            sb_max_excess = -float('inf')

            # Build miniblocks within the superblock
            num_miniblocks = (end - start + self.miniblock_size - 1) // self.miniblock_size
            for m in range(num_miniblocks):
                mb_start = start + m * self.miniblock_size
                mb_end = min(mb_start + self.miniblock_size, end)
                miniblock = {
                    'start': mb_start,
                    'end': mb_end - 1,
                    'e': 0,
                    'm': float('inf'),
                    'M': -float('inf'),
                    'table': None  # For precomputed RMQ within the miniblock
                }
                miniblock_excesses = []
                mb_initial_excess = current_excess
                mb_min_excess = float('inf')
                mb_max_excess = -float('inf')
                for idx in range(mb_start, mb_end):
                    bit = self.bitvector.B[idx]
                    current_excess += 1 if bit == 1 else -1
                    relative_excess = current_excess - mb_initial_excess
                    miniblock_excesses.append(relative_excess)
                    if relative_excess < mb_min_excess:
                        mb_min_excess = relative_excess
                    if relative_excess > mb_max_excess:
                        mb_max_excess = relative_excess
                miniblock['e'] = current_excess - mb_initial_excess
                miniblock['m'] = mb_min_excess
                miniblock['M'] = mb_max_excess
                miniblock['excesses'] = miniblock_excesses
                superblock['miniblocks'].append(miniblock)

                if (mb_min_excess + mb_initial_excess - sb_initial_excess) < sb_min_excess:
                    sb_min_excess = mb_min_excess + mb_initial_excess - sb_initial_excess
                if (mb_max_excess + mb_initial_excess - sb_initial_excess) > sb_max_excess:
                    sb_max_excess = mb_max_excess + mb_initial_excess - sb_initial_excess

            superblock['e'] = current_excess - sb_initial_excess
            superblock['m'] = sb_min_excess
            superblock['M'] = sb_max_excess
            self.superblocks.append(superblock)

    def precompute_miniblock_tables(self):
        """Precompute RMQ and RMQmax results for all possible miniblock configurations."""
        # Since miniblocks are small, we can precompute RMQ results for each unique pattern
        self.miniblock_table = {}
        for superblock in self.superblocks:
            for miniblock in superblock['miniblocks']:
                pattern = tuple(miniblock['excesses'])
                if pattern not in self.miniblock_table:
                    rmq_table, rmqmax_table, mincount_table = self.build_miniblock_tables(pattern)
                    self.miniblock_table[pattern] = (rmq_table, rmqmax_table, mincount_table)
                miniblock['rmq_table'], miniblock['rmqmax_table'], miniblock['mincount_table'] = self.miniblock_table[pattern]

    def build_miniblock_tables(self, excesses):
        """Build RMQ, RMQmax, and mincount tables for a given miniblock excess pattern."""
        n = len(excesses)
        rmq_table = [[-1 for _ in range(n)] for _ in range(n)]
        rmqmax_table = [[-1 for _ in range(n)] for _ in range(n)]
        mincount_table = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            min_value = excesses[i]
            min_index = i
            max_value = excesses[i]
            max_index = i
            min_count = 1
            rmq_table[i][i] = i
            rmqmax_table[i][i] = i
            mincount_table[i][i] = 1
            for j in range(i + 1, n):
                if excesses[j] < min_value:
                    min_value = excesses[j]
                    min_index = j
                    min_count = 1
                elif excesses[j] == min_value:
                    min_count += 1
                if excesses[j] > max_value:
                    max_value = excesses[j]
                    max_index = j
                rmq_table[i][j] = min_index
                rmqmax_table[i][j] = max_index
                mincount_table[i][j] = min_count
        return rmq_table, rmqmax_table, mincount_table

    def get_excess_at_position(self, idx):
        """Compute excess at position idx."""
        if idx < 0:
            return 0
        rank1 = self.bitvector.rank1(idx)
        excess = 2 * rank1 - (idx + 1)
        return excess

    def rmq(self, i, j):
        """Range Minimum Query from index i to j."""
        return self._range_query(i, j, query_type='min')

    def rMq(self, i, j):
        """Range Maximum Query from index i to j."""
        return self._range_query(i, j, query_type='max')

    def _range_query(self, i, j, query_type='min'):
        """General method for range minimum and maximum queries."""
        n = self.n
        if i < 0 or j >= n or i > j:
            raise IndexError("Invalid indices for range query.")

        if query_type == 'min':
            compare = lambda a, b: a < b
            initial_value = float('inf')
        else:
            compare = lambda a, b: a > b
            initial_value = -float('inf')

        best_value = initial_value
        best_index = -1

        initial_excess = self.get_excess_at_position(i - 1) if i > 0 else 0

        # Find starting superblock and miniblock
        start_superblock = i // self.superblock_size
        end_superblock = j // self.superblock_size

        current_excess = initial_excess

        if start_superblock == end_superblock:
            # Query within a single superblock
            idx, val = self._range_query_within_superblock(i, j, current_excess, query_type)
            return idx
        else:
            # Check start superblock
            idx1, val1 = self._range_query_within_superblock(i, (start_superblock + 1) * self.superblock_size - 1, current_excess, query_type)
            if idx1 != -1 and compare(val1, best_value):
                best_value = val1
                best_index = idx1
            # Update current_excess
            current_excess += self.superblocks[start_superblock]['e']
            # Check middle superblocks
            for s in range(start_superblock + 1, end_superblock):
                sb = self.superblocks[s]
                sb_value = (sb['m'] + current_excess) if query_type == 'min' else (sb['M'] + current_excess)
                if compare(sb_value, best_value):
                    best_value = sb_value
                    # Need to find exact index within superblock
                    idx_sb, val_sb = self._range_query_within_superblock(sb['start'], sb['end'], current_excess - sb['e'], query_type)
                    if idx_sb != -1:
                        best_index = idx_sb
                current_excess += sb['e']
            # Check end superblock
            idx2, val2 = self._range_query_within_superblock(end_superblock * self.superblock_size, j, current_excess, query_type)
            if idx2 != -1 and compare(val2, best_value):
                best_value = val2
                best_index = idx2

            return best_index

    def _range_query_within_superblock(self, i, j, initial_excess, query_type='min'):
        """Range query within a superblock."""
        superblock_idx = i // self.superblock_size
        superblock = self.superblocks[superblock_idx]
        start_miniblock = (i - superblock['start']) // self.miniblock_size
        end_miniblock = (j - superblock['start']) // self.miniblock_size

        if query_type == 'min':
            compare = lambda a, b: a < b
            initial_value = float('inf')
        else:
            compare = lambda a, b: a > b
            initial_value = -float('inf')

        best_value = initial_value
        best_index = -1
        current_excess = initial_excess

        if start_miniblock == end_miniblock:
            # Query within a single miniblock
            idx, val = self._range_query_within_miniblock(superblock, start_miniblock, i, j, current_excess, query_type)
            return idx, val
        else:
            # Check start miniblock
            idx1, val1 = self._range_query_within_miniblock(superblock, start_miniblock, i, (start_miniblock + 1) * self.miniblock_size - 1 + superblock['start'], current_excess, query_type)
            if idx1 != -1 and compare(val1, best_value):
                best_value = val1
                best_index = idx1
            # Update current_excess
            current_excess += superblock['miniblocks'][start_miniblock]['e']
            # Check middle miniblocks
            for m in range(start_miniblock + 1, end_miniblock):
                mb = superblock['miniblocks'][m]
                mb_value = (mb['m'] + current_excess) if query_type == 'min' else (mb['M'] + current_excess)
                if compare(mb_value, best_value):
                    best_value = mb_value
                    # Need to find exact index within miniblock
                    idx_mb, val_mb = self._range_query_within_miniblock(superblock, m, mb['start'], mb['end'], current_excess - mb['e'], query_type)
                    if idx_mb != -1:
                        best_index = idx_mb
                current_excess += mb['e']
            # Check end miniblock
            idx2, val2 = self._range_query_within_miniblock(superblock, end_miniblock, end_miniblock * self.miniblock_size + superblock['start'], j, current_excess, query_type)
            if idx2 != -1 and compare(val2, best_value):
                best_value = val2
                best_index = idx2

            return best_index, best_value

    def _range_query_within_miniblock(self, superblock, miniblock_idx, i, j, initial_excess, query_type='min'):
        """Range query within a miniblock using precomputed tables."""
        miniblock = superblock['miniblocks'][miniblock_idx]
        offset_i = i - miniblock['start']
        offset_j = j - miniblock['start']
        if offset_i < 0 or offset_j >= len(miniblock['excesses']):
            return -1, float('inf') if query_type == 'min' else -float('inf')
        if query_type == 'min':
            rmq_table = miniblock['rmq_table']
            min_index = rmq_table[offset_i][offset_j]
            min_value = miniblock['excesses'][min_index] + initial_excess
            absolute_index = miniblock['start'] + min_index
            return absolute_index, min_value
        else:
            rmqmax_table = miniblock['rmqmax_table']
            max_index = rmqmax_table[offset_i][offset_j]
            max_value = miniblock['excesses'][max_index] + initial_excess
            absolute_index = miniblock['start'] + max_index
            return absolute_index, max_value

    def mincount(self, i, j):
        """Count the number of times the minimum excess occurs between positions i and j."""
        min_index = self.rmq(i, j)
        if min_index == -1:
            return 0
        min_value = self.get_excess_at_position(min_index)
        return self._count_excess(i, j, min_value)

    def _count_excess(self, i, j, target_value):
        """Count the number of times a specific excess value occurs between positions i and j."""
        count = 0
        current_excess = self.get_excess_at_position(i - 1) if i > 0 else 0
        for idx in range(i, j + 1):
            bit = self.bitvector.B[idx]
            current_excess += 1 if bit == 1 else -1
            if current_excess == target_value:
                count += 1
        return count

    def minselect(self, i, j, k):
        """Find the position of the k-th occurrence of the minimum excess between positions i and j."""
        min_value = self.get_excess_at_position(self.rmq(i, j))
        count = 0
        current_excess = self.get_excess_at_position(i - 1) if i > 0 else 0
        for idx in range(i, j + 1):
            bit = self.bitvector.B[idx]
            current_excess += 1 if bit == 1 else -1
            if current_excess == min_value:
                count += 1
                if count == k:
                    return idx
        return None

    def fwdsearch(self, i, d):
        """Find the smallest j > i such that excess(j) = excess(i) + d."""
        target_excess = self.get_excess_at_position(i) + d
        n = self.n
        current_excess = self.get_excess_at_position(i)
        idx = i + 1
        while idx < n:
            bit = self.bitvector.B[idx]
            current_excess += 1 if bit == 1 else -1
            if current_excess == target_excess:
                return idx
            idx += 1
        return -1

    def bwdsearch(self, i, d):
        """Find the largest j < i such that excess(j) = excess(i) + d."""
        target_excess = self.get_excess_at_position(i) + d
        current_excess = self.get_excess_at_position(i)
        idx = i - 1
        while idx >= 0:
            bit = self.bitvector.B[idx]
            current_excess -= 1 if bit == 1 else -1
            if current_excess == target_excess:
                return idx
            idx -= 1
        return -1


import unittest
import numpy as np

class SimpleBitVectorTest(unittest.TestCase):
    def setUp(self):
        self.simple_bitvector = [1, 0, 1, 0, 1, 0]
        bitvector = BitVector(self.simple_bitvector)
        bp_tree = BP_Tree(bitvector)
        self.tree = RMMinMaxTree(bp_tree)

    def test_rmq(self):
        # Test rmq(0, 5)
        self.assertEqual(self.tree.rmq(0, 5), 1, "rmq(0, 5) should return 1")

    def test_rMq(self):
        # Test rMq(0, 5)
        self.assertEqual(self.tree.rMq(0, 5), 0, "rMq(0, 5) should return 0")

    def test_mincount(self):
        # Test mincount(0, 5)
        self.assertEqual(self.tree.mincount(0, 5), 3, "mincount(0, 5) should return 3")

    def test_minselect(self):
        # Test minselect(0, 5, 2)
        self.assertEqual(self.tree.minselect(0, 5, 2), 3, "minselect(0, 5, 2) should return 3")


class PeakBitVectorTest(unittest.TestCase):
    def setUp(self):
        self.peak_bitvector = [1, 1, 1, 0, 0, 0]
        bitvector = BitVector(self.peak_bitvector)
        bp_tree = BP_Tree(bitvector)
        self.tree = RMMinMaxTree(bp_tree)

    def test_rmq(self):
        self.assertEqual(self.tree.rmq(0, 5), 5, "rmq(0, 5) should return 5")

    def test_rMq(self):
        self.assertEqual(self.tree.rMq(0, 5), 2, "rMq(0, 5) should return 2")

    def test_mincount(self):
        self.assertEqual(self.tree.mincount(0, 5), 1, "mincount(0, 5) should return 1")

    def test_minselect(self):
        self.assertEqual(self.tree.minselect(0, 5, 1), 5, "minselect(0, 5, 1) should return 5")


class MultipleMinimaBitVectorTest(unittest.TestCase):
    def setUp(self):
        self.multiple_minima_bitvector = [1, 0, 1, 0, 1, 0, 1, 0]
        bitvector = BitVector(self.multiple_minima_bitvector)
        bp_tree = BP_Tree(bitvector)
        self.tree = RMMinMaxTree(bp_tree)

    def test_rmq(self):
        self.assertEqual(self.tree.rmq(0, 7), 1, "rmq(0, 7) should return 1")

    def test_mincount(self):
        self.assertEqual(self.tree.mincount(0, 7), 4, "mincount(0, 7) should return 4")

    def test_minselect(self):
        self.assertEqual(self.tree.minselect(0, 7, 3), 5, "minselect(0, 7, 3) should return 5")


if __name__ == '__main__':
    unittest.main()
