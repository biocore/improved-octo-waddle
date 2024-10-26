import math
import unittest
#from bitarray import bitarray

class BitVector:
    def __init__(self, bitstring):
        # Convert the input bitstring to list of ints
        self.B = [int(b) for b in bitstring]
        #self.B = bitarray(bitstring)
        self.n = len(self.B)  # Total number of bits in the bitvector
        self.lg_n = int(math.log2(self.n)) if self.n > 0 else 1  # Used for calculations
        
        # Define block sizes for rank/select structures
        self.superblock_size = 512  # Size of a superblock
        self.subblock_size = 16     # Size of a subblock within a superblock
        # Variable block sizes?

        # Initialize structures for rank1 (number of 1s up to a position)
        self.rank1_superblock = []  # Cumulative counts at superblock boundaries
        self.rank1_subblock = []    # Cumulative counts at subblock boundaries within superblocks

        # Initialize structures for rank10 (number of '10' patterns up to a position)
        self.rank10_superblock = []  # Cumulative counts at superblock boundaries
        self.rank10_subblock = []    # Cumulative counts at subblock boundaries within superblocks

        # Precompute structures for rank and select operations
        self.precompute_rank_structures()
        # Removed the call to precompute_select_structures()

    def precompute_rank_structures(self):
        total_ones = 0
        total_ten_patterns = 0

        self.rank1_superblock = []
        self.rank10_superblock = []
        self.rank1_subblock = []
        self.rank10_subblock = []

        subblock_ones = 0
        subblock_ten_patterns = 0

        for i in range(self.n):
            # At superblock boundaries
            if i % self.superblock_size == 0:
                self.rank1_superblock.append(total_ones)
                self.rank10_superblock.append(total_ten_patterns)

            # At subblock boundaries
            if i % self.subblock_size == 0 and i != 0:
                self.rank1_subblock.append(subblock_ones)
                self.rank10_subblock.append(subblock_ten_patterns)
                subblock_ones = 0
                subblock_ten_patterns = 0

            bit = self.B[i]
            total_ones += bit
            subblock_ones += bit

            if i > 0 and self.B[i - 1] == 1 and self.B[i] == 0:
                total_ten_patterns += 1
                subblock_ten_patterns += 1

        # Append counts for the last subblock
        self.rank1_subblock.append(subblock_ones)
        self.rank10_subblock.append(subblock_ten_patterns)

        # Store total counts
        self.total_ones = total_ones
        self.total_ten_patterns = total_ten_patterns

    def rank1(self, i):
        if i < 0:
            return 0
        if i >= self.n:
            return self.total_ones

        superblock_idx = i // self.superblock_size
        subblock_idx = i // self.subblock_size

        # Start with the count from the superblock
        rank = self.rank1_superblock[superblock_idx]

        # Add counts from subblocks within the superblock
        subblock_start_idx = (superblock_idx * self.superblock_size) // self.subblock_size
        rank += sum(self.rank1_subblock[subblock_start_idx:subblock_idx])

        # Scan the remaining bits within the subblock
        start_idx = subblock_idx * self.subblock_size
        for idx in range(start_idx, i + 1):
            rank += self.B[idx]

        return rank

    def rank10(self, i):
        if i < 1:
            return 0
        if i >= self.n:
            return self.total_ten_patterns

        superblock_idx = i // self.superblock_size
        subblock_idx = i // self.subblock_size

        # Start with the count from the superblock
        rank = self.rank10_superblock[superblock_idx]

        # Add counts from subblocks within the superblock
        subblock_start_idx = (superblock_idx * self.superblock_size) // self.subblock_size
        rank += sum(self.rank10_subblock[subblock_start_idx:subblock_idx])

        # Scan the remaining bits within the subblock
        start_idx = max(subblock_idx * self.subblock_size, 1)
        for idx in range(start_idx, i + 1):
            if self.B[idx - 1] == 1 and self.B[idx] == 0:
                rank += 1
        return rank

    def select1(self, k):
        if k <= 0 or k > self.total_ones:
            raise ValueError("k is out of bounds")

        # Step 1: Binary search over superblocks
        left, right = 0, len(self.rank1_superblock) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.rank1_superblock[mid] < k:
                left = mid + 1
            else:
                right = mid - 1
        superblock_idx = right

        # Adjust k to be relative to the superblock
        k -= self.rank1_superblock[superblock_idx]

        # Step 2: Linear search over subblocks within the superblock
        superblock_start = superblock_idx * self.superblock_size
        num_subblocks = (min(self.n - superblock_start, self.superblock_size) + self.subblock_size - 1) // self.subblock_size
        subblock_start_idx = (superblock_start) // self.subblock_size

        for i in range(num_subblocks):
            subblock_idx = subblock_start_idx + i
            ones_in_subblock = self.rank1_subblock[subblock_idx]
            if ones_in_subblock >= k:
                # Step 3: Scan within the subblock
                start_idx = superblock_start + i * self.subblock_size
                end_idx = min(start_idx + self.subblock_size, self.n)
                count = 0
                for idx in range(start_idx, end_idx):
                    if self.B[idx] == 1:
                        count += 1
                        if count == k:
                            return idx
                raise ValueError("k is out of bounds within subblock")
            else:
                k -= ones_in_subblock

        # If not found in subblocks, scan remaining bits in the superblock
        start_idx = superblock_start + num_subblocks * self.subblock_size
        end_idx = min(superblock_start + self.superblock_size, self.n)
        count = 0
        for idx in range(start_idx, end_idx):
            if self.B[idx] == 1:
                count += 1
                if count == k:
                    return idx

        raise ValueError("k is out of bounds after scanning subblocks")

    def select10(self, k):
        if k <= 0 or k > self.total_ten_patterns:
            raise ValueError("k is out of bounds")

        # Step 1: Binary search over superblocks
        left, right = 0, len(self.rank10_superblock) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.rank10_superblock[mid] < k:
                left = mid + 1
            else:
                right = mid - 1
        superblock_idx = right

        # Adjust k to be relative to the superblock
        k -= self.rank10_superblock[superblock_idx]

        # Step 2: Linear search over subblocks within the superblock
        superblock_start = superblock_idx * self.superblock_size
        num_subblocks = (min(self.n - superblock_start, self.superblock_size) + self.subblock_size - 1) // self.subblock_size
        subblock_start_idx = (superblock_start) // self.subblock_size

        for i in range(num_subblocks):
            subblock_idx = subblock_start_idx + i
            tens_in_subblock = self.rank10_subblock[subblock_idx]
            if tens_in_subblock >= k:
                # Step 3: Scan within the subblock
                start_idx = max(superblock_start + i * self.subblock_size, 1)
                end_idx = min(start_idx + self.subblock_size, self.n)
                count = 0
                for idx in range(start_idx, end_idx):
                    if self.B[idx - 1] == 1 and self.B[idx] == 0:
                        count += 1
                        if count == k:
                            return idx - 1
                raise ValueError("k is out of bounds within subblock")
            else:
                k -= tens_in_subblock

        # If not found in subblocks, scan remaining bits in the superblock
        start_idx = superblock_start + num_subblocks * self.subblock_size
        end_idx = min(superblock_start + self.superblock_size, self.n)
        count = 0
        for idx in range(max(start_idx, 1), end_idx):
            if self.B[idx - 1] == 1 and self.B[idx] == 0:
                count += 1
                if count == k:
                    return idx - 1

        raise ValueError("k is out of bounds after scanning subblocks")


class TestBitVector(unittest.TestCase):
    def test_basic_rank1_rank10_operations(self):
        """Test Case 1: Basic rank1 and rank10 operations on '1100101'"""
        print("Test Case 1: Basic rank1 and rank10 operations on '1100101'")
        bv = BitVector("1100101")

        # rank1 tests
        self.assertEqual(bv.rank1(0), 1, "rank1(0)")
        self.assertEqual(bv.rank1(3), 2, "rank1(3)")
        self.assertEqual(bv.rank1(6), 4, "rank1(6)")

        # select1 tests
        self.assertEqual(bv.select1(1), 0, "select1(1)")
        self.assertEqual(bv.select1(2), 1, "select1(2)")
        self.assertEqual(bv.select1(4), 6, "select1(4)")

        # rank10 tests
        self.assertEqual(bv.rank10(0), 0, "rank10(0)")
        self.assertEqual(bv.rank10(4), 1, "rank10(4)")
        self.assertEqual(bv.rank10(6), 2, "rank10(6)")

        # select10 tests
        self.assertEqual(bv.select10(1), 1, "select10(1)")
        self.assertEqual(bv.select10(2), 4, "select10(2)")

    def test_empty_bitvector(self):
        """Test Case 2: Edge Case with empty bitstring"""
        print("Test Case 2: Edge Case with empty bitstring")
        bv_empty = BitVector("")
        self.assertEqual(bv_empty.rank1(0), 0, "rank1 on empty bitvector")
        self.assertEqual(bv_empty.rank10(0), 0, "rank10 on empty bitvector")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for empty bitvector"):
            bv_empty.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for empty bitvector"):
            bv_empty.select10(1)
        print("Empty bitvector operations passed.\n")

    def test_all_ones(self):
        """Test Case 3: BitVector with only '1's"""
        print("Test Case 3: BitVector with only '1's")
        bv_ones = BitVector("111111")
        self.assertEqual(bv_ones.rank1(5), 6, "rank1(5) - All Ones")
        self.assertEqual(bv_ones.select1(4), 3, "select1(4) - All Ones")
        self.assertEqual(bv_ones.rank10(5), 0, "rank10(5) - All Ones")
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for all-ones bitvector"):
            bv_ones.select10(1)
        print("All-ones bitvector operations passed.\n")

    def test_all_zeros(self):
        """Test Case 4: BitVector with only '0's"""
        print("Test Case 4: BitVector with only '0's")
        bv_zeros = BitVector("000000")
        self.assertEqual(bv_zeros.rank1(5), 0, "rank1(5) - All Zeros")
        self.assertEqual(bv_zeros.rank10(5), 0, "rank10(5) - All Zeros")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for all-zeros bitvector"):
            bv_zeros.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for all-zeros bitvector"):
            bv_zeros.select10(1)
        print("All-zeros bitvector operations passed.\n")

    def test_alternating_bits(self):
        """Test Case 5: Complex pattern '101010'"""
        print("Test Case 5: Complex pattern '101010'")
        bv_pattern = BitVector("101010")
        self.assertEqual(bv_pattern.rank1(5), 3, "rank1(5) - Alternating")
        self.assertEqual(bv_pattern.rank10(5), 3, "rank10(5) - Alternating")
        self.assertEqual(bv_pattern.select1(2), 2, "select1(2) - Alternating")
        self.assertEqual(bv_pattern.select10(1), 0, "select10(1) - Alternating")
        self.assertEqual(bv_pattern.select10(2), 2, "select10(2) - Alternating")
        with self.assertRaises(ValueError, msg="select10(4) should raise ValueError for '101010' bitvector"):
            bv_pattern.select10(4)
        print("Complex pattern operations passed.\n")

    def test_single_one(self):
        """Test Case 6: Single '1'"""
        print("Test Case 6: Single '1'")
        bv_single_one = BitVector("1")
        self.assertEqual(bv_single_one.rank1(0), 1, "rank1(0) - Single '1'")
        self.assertEqual(bv_single_one.rank10(0), 0, "rank10(0) - Single '1'")
        self.assertEqual(bv_single_one.select1(1), 0, "select1(1) - Single '1'")
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for single '1' bitvector"):
            bv_single_one.select10(1)
        print("Single '1' bitvector operations passed.\n")

    def test_single_zero(self):
        """Test Case 7: Single '0'"""
        print("Test Case 7: Single '0'")
        bv_single_zero = BitVector("0")
        self.assertEqual(bv_single_zero.rank1(0), 0, "rank1(0) - Single '0'")
        self.assertEqual(bv_single_zero.rank10(0), 0, "rank10(0) - Single '0'")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for single '0' bitvector"):
            bv_single_zero.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for single '0' bitvector"):
            bv_single_zero.select10(1)
        print("Single '0' bitvector operations passed.\n")

    def test_out_of_bounds_rank(self):
        """Test Case 8: Out-of-Bounds Indices for `rank`"""
        print("Test Case 8: Out-of-Bounds Indices for `rank`")
        bv = BitVector("1100101")
        self.assertEqual(bv.rank1(10), bv.total_ones, "rank1(10) - Out of Bounds")
        self.assertEqual(bv.rank10(10), bv.total_ten_patterns, "rank10(10) - Out of Bounds")
        print("Out-of-bounds rank operations passed.\n")

    def test_invalid_select(self):
        """Test Case 9: Invalid Occurrences for `select`"""
        print("Test Case 9: Invalid Occurrences for `select`")
        bv = BitVector("1100101")
        with self.assertRaises(ValueError, msg="select1(0) should raise ValueError"):
            bv.select1(0)
        with self.assertRaises(ValueError, msg="select1(5) should raise ValueError"):
            bv.select1(5)
        with self.assertRaises(ValueError, msg="select10(0) should raise ValueError"):
            bv.select10(0)
        with self.assertRaises(ValueError, msg="select10(3) should raise ValueError"):
            bv.select10(3)
        print("Invalid select operations passed.\n")

    def test_large_bitvector_stress(self):
        """Test Case 10: Large BitVector Stress Test"""
        print("Test Case 10: Large BitVector Stress Test")
        large_bv = BitVector("101" * 1000)  # 3000-bit pattern
        self.assertEqual(large_bv.rank1(2999), 2000, "rank1(2999) - Large BitVector")
        self.assertEqual(large_bv.rank10(2999), 1000, "rank10(2999) - Large BitVector")
        self.assertEqual(large_bv.select1(500), 749, "select1(500) - Large BitVector")
        self.assertEqual(large_bv.select10(500), 1497, "select10(500) - Large BitVector")
        print("Large BitVector operations passed.\n")

if __name__ == '__main__':
    unittest.main()
