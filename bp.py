import numpy as np
import math


class RMMTree:
    """
    A complete-binary-tree storage of the rmM-tree for a balanced parentheses sequence.
    Leaves correspond to blocks of size 'b'. Internal nodes merge child info.
    
    Each node i stores:
      e[i] : total excess of node i's range
      m[i] : minimum prefix-excess in node i's range (relative to that node's start)
      M[i] : maximum prefix-excess
      n[i] : number of occurrences of m[i]
    
    Using 1-based indexing:
      - The root is at index 1.
      - The children of node i are 2*i (left) and 2*i+1 (right).
    Leaves are found in [leaf_start..leaf_start + num_leaves - 1].
    """

    def __init__(self, parentheses, b=None):
        """
        :param parentheses: A string of '(' and ')'.
        :param b: Block size for the leaves. If None, we pick b = floor(sqrt(n)) or 1.
        """
        self.parentheses = parentheses
        self.n = len(parentheses)
        if self.n == 0:
            # Edge case: empty string -> no tree
            self.b = 1
            self.tree_size = 0
            self.e = []
            self.m = []
            self.M = []
            self.count_min = []
            return

        # Choose block size
        if b is None:
            b = max(1, int(math.log2(self.n)))
        self.b = b

        # Number of leaf blocks
        self.num_leaves = (self.n + self.b - 1) // self.b

        # Height of the complete binary tree needed to store 'num_leaves' leaves
        # Round up to the next power of two.
        self.height = math.ceil(math.log2(self.num_leaves)) if self.num_leaves > 0 else 0
        self.leaf_count = 2**self.height if self.num_leaves > 0 else 0

        # Total nodes in a perfect binary tree with leaf_count leaves
        self.tree_size = 2*self.leaf_count - 1 if self.leaf_count > 0 else 0

        # Arrays for storing (e, m, M, count_min) in each node
        self.e = [0] * (self.tree_size + 1)
        self.m = [0] * (self.tree_size + 1)
        self.M = [0] * (self.tree_size + 1)
        self.count_min = [0] * (self.tree_size + 1)

        # The leaves occupy indices [leaf_start..leaf_start + leaf_count - 1]
        self.leaf_start = 2**self.height if self.leaf_count > 0 else 0

        if self.n > 0:
            # 1) Build the leaves from the blocks
            self._build_leaves()

            # 2) Build internal nodes by merging
            self._build_internal_nodes()

    # Build the leaves from block summaries
    def _build_leaves(self):
        for block_idx in range(self.num_leaves):
            node_index = self.leaf_start + block_idx

            # Range [start, end) in the parentheses array
            start = block_idx * self.b
            end = min(start + self.b, self.n)

            e_val, m_val, M_val, n_val = self._compute_block_summary(start, end)
            self.e[node_index] = e_val
            self.m[node_index] = m_val
            self.M[node_index] = M_val
            self.count_min[node_index] = n_val

        # Fill dummy leaves if necessary
        for block_idx in range(self.num_leaves, self.leaf_count):
            node_index = self.leaf_start + block_idx
            self.e[node_index] = 0
            self.m[node_index] = 0
            self.M[node_index] = 0
            self.count_min[node_index] = 0 

    def _compute_block_summary(self, start, end):
        current_excess = 0
        m_val = float('inf')
        M_val = float('-inf')
        n_val = 0

        for i in range(start, end):
            if self.parentheses[i] == '(':
                current_excess += 1
            else:
                current_excess -= 1

            if current_excess < m_val:
                m_val = current_excess
                n_val = 1
            elif current_excess == m_val:
                n_val += 1
            if current_excess > M_val:
                M_val = current_excess

        e_val = current_excess
        return (e_val, m_val, M_val, n_val)

    # Build internal nodes by merging children
    def _build_internal_nodes(self):
        for i in range(self.leaf_start - 1, 0, -1):
            left = 2*i
            right = 2*i + 1

            e_left, m_left, M_left, n_left = (
                self.e[left], self.m[left], self.M[left], self.count_min[left]
            )
            e_right, m_right, M_right, n_right = (
                self.e[right], self.m[right], self.M[right], self.count_min[right]
            )

            # parent's e
            e_val = e_left + e_right

            # parent's m
            m_right_adjusted = e_left + m_right
            m_val = min(m_left, m_right_adjusted)

            # parent's M
            M_right_adjusted = e_left + M_right
            M_val = max(M_left, M_right_adjusted)

            # parent's count_min
            if m_left == m_right_adjusted == m_val:
                n_val = n_left + n_right
            elif m_left == m_val:
                n_val = n_left
            else:
                n_val = n_right

            self.e[i] = e_val
            self.m[i] = m_val
            self.M[i] = M_val
            self.count_min[i] = n_val

    # Merge helper for queries
    def _merge_segments(self, segA, segB):
        if segA is None:
            return segB
        if segB is None:
            return segA

        (e_left, m_left, M_left, n_left) = segA
        (e_right, m_right, M_right, n_right) = segB

        e_val = e_left + e_right
        m_right_adj = e_left + m_right
        m_val = min(m_left, m_right_adj)

        M_right_adj = e_left + M_right
        M_val = max(M_left, M_right_adj)

        if m_left == m_right_adj == m_val:
            n_val = n_left + n_right
        elif m_left == m_val:
            n_val = n_left
        else:
            n_val = n_right

        return (e_val, m_val, M_val, n_val)

    # Recursive range query for [query_start..query_end]
    def _range_query(self, index, seg_start, seg_end, query_start, query_end):
        if query_end < seg_start or query_start > seg_end:
            return None

        if query_start <= seg_start and seg_end <= query_end:
            return (self.e[index], self.m[index], self.M[index], self.count_min[index])

        mid = (seg_start + seg_end) // 2
        left_res = self._range_query(2*index, seg_start, mid, query_start, query_end)
        right_res = self._range_query(2*index + 1, mid+1, seg_end, query_start, query_end)

        return self._merge_segments(left_res, right_res)

    # Range-min query
    def range_min_query(self, L_block, R_block):
        if L_block < 0 or R_block >= self.leaf_count or L_block > R_block:
            raise ValueError("Invalid block range.")

        result = self._range_query(1, 0, self.leaf_count - 1, L_block, R_block)
        if result is None:
            return None
        return result[1]  # 'm' is the min prefix-excess

    # Range-max query
    def range_max_query(self, L_block, R_block):
        if L_block < 0 or R_block >= self.leaf_count or L_block > R_block:
            raise ValueError("Invalid block range.")

        result = self._range_query(1, 0, self.leaf_count - 1, L_block, R_block)
        if result is None:
            return None
        return result[2]  # 'M' is the max prefix-excess

    # Compute (e, m, M, n) for any subblock parentheses[start..end-1],
    # with a small linear scan.
    def _compute_range_summary(self, start, end):
        current_excess = 0
        m_val = float('inf')
        M_val = float('-inf')
        n_val = 0

        for i in range(start, end):
            if self.parentheses[i] == '(':
                current_excess += 1
            else:
                current_excess -= 1

            if current_excess < m_val:
                m_val = current_excess
                n_val = 1
            elif current_excess == m_val:
                n_val += 1

            if current_excess > M_val:
                M_val = current_excess

        e_val = current_excess
        return (e_val, m_val, M_val, n_val)

    # Merge full blocks in [block_start..block_end] into a single (e,m,M,n).
    def _range_merge_query(self, block_start, block_end):
        if block_start > block_end:
            return None
        return self._range_query(1, 0, self.leaf_count - 1, block_start, block_end)

    # High-level query for parentheses[L..R], with partial block coverage
    def query_range_characters(self, L, R):
        if L < 0 or R >= self.n or L > R:
            raise ValueError("Invalid character range.")

        L_block = L // self.b
        R_block = R // self.b

        if L_block == R_block:
            return self._compute_range_summary(L, R+1)

        # Partial left block
        left_block_end = (L_block+1) * self.b
        partial_left = self._compute_range_summary(L, left_block_end)

        # Partial right block
        right_block_start = R_block * self.b
        partial_right = self._compute_range_summary(right_block_start, R+1)

        # Fully covered middle blocks
        middle_summary = self._range_merge_query(L_block+1, R_block-1)

        # Merge all parts
        left_merge = self._merge_segments(partial_left, middle_summary)
        full_merge = self._merge_segments(left_merge, partial_right)

        return full_merge

    def get_root_summary(self):
        if self.tree_size == 0:
            return None
        return (self.e[1], self.m[1], self.M[1], self.count_min[1])

    # Computes excess at any position
    def get_excess(self, i):
        """
        Retrieve the prefix excess at position i.
        :param i: 0-based index (0 <= i <= n)
        :return: prefix_excess[i]
        """
        if i < 0 or i > self.n:
            raise ValueError("Index i out of bounds.")

        block_idx = i // self.b
        if block_idx == 0:
            cumulative_e = 0
        else:
            cumulative_e = self.range_e_sum(0, block_idx - 1)

        block_start = block_idx * self.b
        scan_end = min(i+1, self.n)

        # Linear scan within the block to compute excess up to position i
        within_block_e = 0
        for j in range(block_start, scan_end):
            if self.parentheses[j] == '(':
                within_block_e += 1
            else:
                within_block_e -= 1

        return cumulative_e + within_block_e

    def range_e_sum(self, L_block, R_block):
        """
        Sum of 'e' over blocks [L_block, R_block].
        :param L_block: 0-based left block index.
        :param R_block: 0-based right block index.
        :return: sum of 'e' over the range.
        """
        if L_block > R_block:
            return 0
        result = self._range_query(1, 0, self.leaf_count - 1, L_block, R_block)
        if result is None:
            return 0
        return result[0] 

    def print_tree(self):
        print(f"\n=== rmm-tree ===")
        print(f"n = {self.n}, b = {self.b}, num_leaves = {self.num_leaves}")
        print(f"height = {self.height}, leaf_count = {self.leaf_count}, tree_size = {self.tree_size}\n")

        level = 0
        start_index = 1
        while start_index <= self.tree_size:
            end_index = min(start_index*2 - 1, self.tree_size)
            nodes_this_level = list(range(start_index, end_index+1))
            data_str = []
            for i in nodes_this_level:
                data_str.append(
                    f"[i={i}, e={self.e[i]}, m={self.m[i]}, M={self.M[i]}, n={self.count_min[i]}]"
                )
            print(f"Level {level}:\n  " + "  ".join(data_str))
            start_index = end_index+1
            level += 1


class BP:
    """
    Balanced parentheses data structure using RMMTree for range min/max queries.
    """
    def __init__(self, 
                 B, 
                 lengths=None,
                 names=None,
                 edges=None):
        """
        :param B: a NumPy 1D array or list of booleans, where 1='(' and 0=')'
        :param lengths: optional array of floats (lengths per position)
        :param names: optional array of objects (names per position)
        :param edges: optional array of integers (edge references)
        """
        assert np.sum(B) == (float(len(B)) / 2), "Must have an equal number of '(' and ')'."

        self.B = np.array(B, dtype=bool)  # store as a boolean array
        self.size = self.B.size

        # Convert B to parentheses => '(' or ')'
        # TODO: Use np array or bitarray for greater efficiency?
        par_list = []
        for bit in self.B:
            if bit:
                par_list.append('(')
            else:
                par_list.append(')')
        parentheses_str = ''.join(par_list)

        # Create RMMTree
        pick_b = max(1, int(math.log2(self.size)))
        self._rmm = RMMTree(parentheses_str, b=pick_b)

        # names / lengths
        if names is not None:
            self._names = names
        else:
            self._names = np.full(self.size, None, dtype=object)

        if lengths is not None:
            self._lengths = lengths
        else:
            self._lengths = np.zeros(self.size, dtype=float)

        # edges
        if edges is not None:
            self._set_edges(edges)
        else:
            self._edges = np.full(self.size, 0, dtype=int)
            self._edge_lookup = None

        # Precompute for select( t, k )
        # rank(0) = cumsum of (1 - B), rank(1) = cumsum of B
        _r_index_0 = np.cumsum((1 - self.B), dtype=int)
        _r_index_1 = np.cumsum(self.B, dtype=int)
        uniq_idx_0 = np.unique(_r_index_0, return_index=True)[1].astype(int)
        uniq_idx_1 = np.unique(_r_index_1, return_index=True)[1].astype(int)
        self._k_index_0 = uniq_idx_0
        self._k_index_1 = uniq_idx_1

    # Basic Setup
    def write(self, fname):
        """Write structure to a compressed .npz file."""
        np.savez_compressed(fname, 
                            names=self._names, 
                            lengths=self._lengths, 
                            B=self.B)

    @staticmethod
    def read(fname):
        """Read from a .npz file created by BP.write()"""
        data = np.load(fname)
        bp = BP(data['B'], names=data['names'], lengths=data['lengths'])
        return bp

    def set_names(self, names):
        self._names = names

    def set_lengths(self, lengths):
        self._lengths = lengths

    def _set_edges(self, edges):
        """
        Build a lookup so that edge_from_number(n) returns the index i
        where edges[i] == n (for positions that are '(').
        """
        n = len(self.B)
        _edge_lookup = np.full(n, 0, dtype=int)
        for i in range(n):
            if self.B[i]:
                edge = edges[i]
                _edge_lookup[edge] = i
        self._edge_lookup = _edge_lookup
        self._edges = edges

    def set_edges(self, edges):
        self._set_edges(edges)

    def name(self, i):
        return self._names[i]

    def length(self, i):
        return self._lengths[i]

    def edge(self, i):
        return self._edges[i]

    def edge_from_number(self, n):
        return self._edge_lookup[n]

    #  Compute prefix excess / rank / select
    def _excess(self, i):
        if i < 0:
            return 0
        return self._rmm.get_excess(i)

    def rank(self, t, i):
        if i < 0:
            return 0
        if i >= self.size:
            i = self.size - 1
        net = self._rmm.get_excess(i)
        opens_up_to_i = (net + (i+1)) // 2
        if t == 1:
            return opens_up_to_i
        else:
            return (i+1) - opens_up_to_i

    def select(self, t, k):
        # Using the precomputed arrays
        if t == 1:
            return self._k_index_1[k]
        else:
            return self._k_index_0[k]

    #     Queries that rely on get_excess
    def close(self, i):
        if not self.B[i]:
            return i  # already ')'
        return self.fwdsearch(i, -1)

    def open(self, i):
        if self.B[i] or i <= 0:
            return i
        return self.bwdsearch(i, 0) + 1

    def enclose(self, i):
        if self.B[i]:
            return self.bwdsearch(i, -2) + 1
        else:
            return self.bwdsearch(i-1, -2) + 1

    # Naive forward/backward search
    # def fwdsearch(self, i, d):
    #     if i >= self.size:
    #         return -1
    #     start_ex = self._rmm.get_excess(i)
    #     goal = start_ex + d
    #     for j in range(i+1, self.size):
    #         if self._rmm.get_excess(j) == goal:
    #             return j
    #     return -1

    # def bwdsearch(self, i, d):
    #     if i < 0:
    #         return -1
    #     start_ex = self._rmm.get_excess(i)
    #     goal = start_ex + d
    #     for j in range(i-1, -1, -1):
    #         if self._rmm.get_excess(j) == goal:
    #             return j
    #     return -1
    
    # Optimized forward/backward search
    def fwdsearch(self, i, d):
        """
        Return the smallest j > i such that _rmm.get_excess(j) == _rmm.get_excess(i) + d.
        This version uses a block‐skipping strategy.
        Returns None if not found.
        """
        # Use self.size (which you set in BP) or self._rmm.n if that is your canonical length.
        if i >= self.size:
            return -1
        target = self._rmm.get_excess(i) + d

        # Phase 1: Scan remainder of current block.
        block_idx = i // self._rmm.b
        block_start = block_idx * self._rmm.b
        block_end = min(block_start + self._rmm.b, self.size)
        for j in range(i+1, block_end):
            if self._rmm.get_excess(j) == target:
                return j

        # Phase 2: For each subsequent block, use the leaf summaries.
        # We adjust the block’s local min/max by computing the offset
        # as the global excess immediately before the block.
        for k in range(block_idx+1, self._rmm.num_leaves):
            offset = self._rmm.get_excess(k * self._rmm.b - 1) if k > 0 else 0
            local_m = self._rmm.m[self._rmm.leaf_start + k]
            local_M = self._rmm.M[self._rmm.leaf_start + k]
            global_m = offset + local_m
            global_M = offset + local_M
            if global_m <= target <= global_M:
                b_start = k * self._rmm.b
                b_end = min(b_start + self._rmm.b, self.size)
                for j in range(b_start, b_end):
                    if self._rmm.get_excess(j) == target:
                        return j
        return -1


    def bwdsearch(self, i, d):
        """
        Return the largest j < i such that _rmm.get_excess(j) == _rmm.get_excess(i) + d.
        This version uses a block‐skipping strategy.
        Returns None if not found.
        """
        if i < 0:
            return -1
        target = self._rmm.get_excess(i) + d

        # Phase 1: Scan backward in current block.
        block_idx = (i - 1) // self._rmm.b
        block_start = block_idx * self._rmm.b
        for j in range(i - 1, block_start - 1, -1):
            if self._rmm.get_excess(j) == target:
                return j

        # Phase 2: For each preceding block, check the leaf summary.
        for k in range(block_idx - 1, -1, -1):
            offset = self._rmm.get_excess(k * self._rmm.b - 1) if k > 0 else 0
            local_m = self._rmm.m[self._rmm.leaf_start + k]
            local_M = self._rmm.M[self._rmm.leaf_start + k]
            global_m = offset + local_m
            global_M = offset + local_M
            if global_m <= target <= global_M:
                b_start = k * self._rmm.b
                b_end = min(b_start + self._rmm.b, self.size)
                # Scan backward through this block.
                for j in range(b_end - 1, b_start - 1, -1):
                    if self._rmm.get_excess(j) == target:
                        return j
        return -1

    #   Optimized rmq / rMq with rmM-tree
    def rmq(self, i, j):
        """
        Return the index of the leftmost minimum prefix-excess in [i..j].
        Uses partial-block coverage plus _rmm range queries, O(log(#blocks)).
        """
        if j < i:
            return -1
        n = self.size
        if i < 0: i = 0
        if j >= n: j = n-1

        # (1) Find global min value in [i..j]
        min_val = self._range_min_value(i, j)
        # (2) Find the leftmost occurrence of min_val in [i..j]
        return self._range_min_select(i, j, min_val, q=1)

    def rMq(self, i, j):
        """
        Return the index of the leftmost maximum prefix-excess in [i..j].
        Uses partial-block coverage plus _rmm range queries, O(log(#blocks)).
        """
        if j < i:
            return -1
        n = self.size
        if i < 0: i = 0
        if j >= n: j = n-1

        max_val = self._range_max_value(i, j)
        return self._range_max_select(i, j, max_val, q=1)

    def _range_min_value(self, L, R):
        """
        Return the minimum prefix-excess in [L..R], in O(log(#blocks)) 
        using partial coverage + self._rmm._range_merge_query.
        """
        if L > R:
            return float('inf')
        # The rmM-tree divides the parentheses string into blocks of size b
        b = self._rmm.b
        Lb = L // b
        Rb = R // b

        # partial left coverage
        left_val = float('inf')
        left_idx_end = min((Lb+1)*b, self.size)
        if Lb == Rb:
            # fully in one block => do direct partial scan
            return self._compute_local_min(L, R+1)

        # partial left block
        if L < left_idx_end:
            left_val = self._compute_local_min(L, left_idx_end)

        # partial right block
        right_val = float('inf')
        right_idx_start = Rb * b
        if right_idx_start <= R and Rb > Lb:
            right_val = self._compute_local_min(right_idx_start, R+1)

        # middle coverage from blocks [Lb+1..Rb-1]
        mid_val = float('inf')
        if Rb - 1 >= Lb + 1:
            seg = self._rmm._range_merge_query(Lb+1, Rb-1)  
            if seg is not None:
                e_val, m_val, M_val, n_val = seg
                offset = self._rmm.get_excess((Lb+1)*b - 1) if ((Lb+1)*b - 1) >= 0 else 0
                mid_val = offset + m_val

        return min(left_val, right_val, mid_val)

    def _range_max_value(self, L, R):
        """
        Return the maximum prefix-excess in [L..R], using partial coverage + merges.
        """
        if L > R:
            return float('-inf')
        b = self._rmm.b
        Lb = L // b
        Rb = R // b

        left_val = float('-inf')
        left_idx_end = min((Lb+1)*b, self.size)
        if Lb == Rb:
            return self._compute_local_max(L, R+1)

        if L < left_idx_end:
            left_val = self._compute_local_max(L, left_idx_end)

        right_val = float('-inf')
        right_idx_start = Rb * b
        if right_idx_start <= R and Rb > Lb:
            right_val = self._compute_local_max(right_idx_start, R+1)

        mid_val = float('-inf')
        if Rb - 1 >= Lb + 1:
            seg = self._rmm._range_merge_query(Lb+1, Rb-1)  # => (e,m,M,n)
            if seg is not None:
                e_val, m_val, M_val, n_val = seg
                offset = self._rmm.get_excess((Lb+1)*b - 1) if ((Lb+1)*b - 1) >= 0 else 0
                mid_val = offset + M_val


        return max(left_val, right_val, mid_val)

    def _range_min_select(self, L, R, min_val, q):
        """
        Find the leftmost position of the q-th occurrence of 'min_val' in [L..R].
        Partial coverage in left block, right block, plus middle blocks 
        with a top-down approach if needed.
        For simplicity, we do a linear pass in the partial blocks,
        then a top-down approach in the middle blocks.
        """
        b = self._rmm.b
        Lb = L // b
        Rb = R // b

        # 1) partial left block
        if Lb == Rb:
            return self._local_min_select(L, R+1, min_val, q)
        else:
            left_idx_end = min((Lb+1)*b, self.size)
            # partial coverage [L..left_idx_end-1]
            count_left = self._count_local_min(L, left_idx_end, min_val)
            if q <= count_left:
                # found inside partial left
                return self._local_min_select(L, left_idx_end, min_val, q)
            q -= count_left

        # 2) middle blocks
        if Rb - 1 >= Lb + 1:
            # we do a range query for blocks [Lb+1..Rb-1], 
            # if there's a min in that coverage, we do a top-down approach to find the block.
            seg = self._rmm._range_merge_query(Lb+1, Rb-1)
            if seg is not None:
                e_val, m_val, M_val, n_val = seg
                offset = self._rmm.get_excess((Lb+1)*b - 1) if ((Lb+1)*b - 1) >= 0 else 0
                # if m_val == min_val:
                if offset + m_val == min_val:
                    # that means the middle coverage contains occurrences
                    # if q <= n_val => it's in there
                    if q <= n_val:
                        # top-down to find the block
                        return self._range_min_select_in_blocks(Lb+1, Rb-1, min_val, q)
                    else:
                        q -= n_val

        # 3) partial right block
        right_idx_start = Rb * b
        if right_idx_start <= R:
            count_right = self._count_local_min(right_idx_start, R+1, min_val)
            if q <= count_right:
                return self._local_min_select(right_idx_start, R+1, min_val, q)
            else:
                return None  # not found

        return None

    def _range_max_select(self, L, R, max_val, q):
        """
        Mirroring _range_min_select, but for maximum.
        """
        b = self._rmm.b
        Lb = L // b
        Rb = R // b

        # partial left
        if Lb == Rb:
            return self._local_max_select(L, R+1, max_val, q)
        else:
            left_end = min((Lb+1)*b, self.size)
            count_left = self._count_local_max(L, left_end, max_val)
            if q <= count_left:
                return self._local_max_select(L, left_end, max_val, q)
            q -= count_left

        # middle
        if Rb - 1 >= Lb + 1:
            seg = self._rmm._range_merge_query(Lb+1, Rb-1)
            if seg is not None:
                e_val, m_val, M_val, n_val = seg
                offset = self._rmm.get_excess((Lb+1)*b - 1) if ((Lb+1)*b - 1) >= 0 else 0
                # if M_val == max_val:
                if offset + M_val == max_val:
                    # we must get the #occurrences of max_val in these blocks
                    # but that's not directly stored. We'll do a top-down approach 
                    # that also counts how many times M_val appears. 
                    # For simplicity we can do an approach similar to min. 
                    count_mid = self._count_blocks_value(Lb+1, Rb-1, max_val, is_min=False)
                    if q <= count_mid:
                        return self._range_max_select_in_blocks(Lb+1, Rb-1, max_val, q)
                    else:
                        q -= count_mid

        # partial right
        right_start = Rb*b
        if right_start <= R:
            count_right = self._count_local_max(right_start, R+1, max_val)
            if q <= count_right:
                return self._local_max_select(right_start, R+1, max_val, q)
            else:
                return None
        return None

    # local partial scans for min / max
    def _compute_local_min(self, start, end):
        """
        Return the min prefix-excess in [start..end-1].
        (small linear scan, good for partial block)
        """
        val = float('inf')
        for pos in range(start, end):
            ex = self._rmm.get_excess(pos)
            if ex < val:
                val = ex
        return val

    def _compute_local_max(self, start, end):
        mx = float('-inf')
        for pos in range(start, end):
            ex = self._rmm.get_excess(pos)
            if ex > mx:
                mx = ex
        return mx

    def _count_local_min(self, start, end, min_val):
        cnt = 0
        for pos in range(start, end):
            if self._rmm.get_excess(pos) == min_val:
                cnt += 1
        return cnt

    def _count_local_max(self, start, end, max_val):
        cnt = 0
        for pos in range(start, end):
            if self._rmm.get_excess(pos) == max_val:
                cnt += 1
        return cnt

    def _local_min_select(self, start, end, min_val, q):
        """
        find the qth occurrence of min_val in [start..end-1], left to right
        """
        found = 0
        for pos in range(start, end):
            if self._rmm.get_excess(pos) == min_val:
                found += 1
                if found == q:
                    return pos
        return None

    def _local_max_select(self, start, end, max_val, q):
        found = 0
        for pos in range(start, end):
            if self._rmm.get_excess(pos) == max_val:
                found += 1
                if found == q:
                    return pos
        return None

    # top-down approach for min/max inside [block_start..block_end] in rmM-tree
    def _range_min_select_in_blocks(self, blockL, blockR, min_val, q):
        """
        We know the global min in blocks [blockL..blockR] is min_val, 
        and there are at least q occurrences. 
        We'll do a partial top-down approach. 
        Simpler approach: we do a "count and skip" from left to right block. 
        This is effectively a micro-implementation of minselect. 
        """
        # We'll iterate block by block (not great if many blocks, but simpler).
        # For a fully optimized approach, you do a segment-tree "select" approach.
        for blk in range(blockL, blockR+1):
            seg = self._rmm._range_query(1, 0, self._rmm.leaf_count-1, blk, blk)
            if seg is None:
                continue
            e_val, m_val, M_val, n_val = seg
            offset = self._rmm.get_excess(blk * self._rmm.b - 1) if (blk * self._rmm.b - 1) >= 0 else 0
            # if m_val > min_val:
            if offset + m_val > min_val:
                # no occurrences
                continue
            else:
                # there's at least n_val occurrences if m_val==min_val
                # partial block range => we can do a local partial scan 
                # from block_start..block_end in the original string
                block_start = blk * self._rmm.b
                block_end = min(block_start + self._rmm.b, self.size)
                # count how many times min_val appears
                c = self._count_local_min(block_start, block_end, min_val)
                if q <= c:
                    return self._local_min_select(block_start, block_end, min_val, q)
                else:
                    q -= c
        return None

    def _range_max_select_in_blocks(self, blockL, blockR, max_val, q):
        """
        Mirroring the min approach, but for max_val
        """
        for blk in range(blockL, blockR+1):
            seg = self._rmm._range_query(1, 0, self._rmm.leaf_count-1, blk, blk)
            if seg is None:
                continue
            e_val, m_val, M_val, n_val = seg
            offset = self._rmm.get_excess(blk * self._rmm.b - 1) if (blk * self._rmm.b - 1) >= 0 else 0
            # if M_val < max_val:
            if offset + M_val < max_val:
                continue
            # there's an occurrence in this block
            block_start = blk * self._rmm.b
            block_end = min(block_start + self._rmm.b, self.size)
            c = self._count_local_max(block_start, block_end, max_val)
            if q <= c:
                return self._local_max_select(block_start, block_end, max_val, q)
            else:
                q -= c
        return None

    def _count_blocks_value(self, blockL, blockR, val, is_min=True):
        """
        Count occurrences of 'val' across blocks [blockL..blockR].
        For min or max, same logic, we do partial scanning of each block.
        """
        total = 0
        for blk in range(blockL, blockR+1):
            block_start = blk * self._rmm.b
            block_end = min(block_start + self._rmm.b, self.size)
            if is_min:
                total += self._count_local_min(block_start, block_end, val)
            else:
                total += self._count_local_max(block_start, block_end, val)
        return total

    # The rest: parent, child, lca, shear, etc.
    # unchanged except we now have better rmq/rMq
    def __len__(self):
        return self.size // 2

    def __repr__(self):
        total_nodes = len(self)
        tip_count = self.ntips()
        return f"<BP, name: {self.name(0)}, internal node count: {total_nodes - tip_count}, tips count: {tip_count}>"

    def __reduce__(self):
        return (BP, (self.B, self._lengths, self._names))

    def depth(self, i):
        return self._rmm.get_excess(i)

    def root(self):
        return 0

    def parent(self, i):
        if i == self.root() or i == (self.size - 1):
            return -1
        return self.enclose(i)

    def isleaf(self, i):
        if i+1 >= self.size:
            return False
        return self.B[i] and (not self.B[i+1])

    def fchild(self, i):
        if self.B[i]:
            if self.isleaf(i):
                return 0
            else:
                return i+1
        else:
            return self.fchild(self.open(i))

    def lchild(self, i):
        if self.B[i]:
            if self.isleaf(i):
                return 0
            else:
                return self.open(self.close(i) - 1)
        else:
            return self.lchild(self.open(i))

    def mincount(self, i, j):
        """
        # of occurrences of the minimum in excess(i..j).
        We'll reuse rmq-based logic: find min_val, then count how many times it appears.
        This is O((j-i)/b + log(#blocks)) but we can refine if we want.
        """
        if j < i:
            return 0
        min_val = self._range_min_value(i, j)
        # we can do a second pass to count occurrences, as above:
        b = self._rmm.b
        Lb = i // b
        Rb = j // b
        cnt = 0
        # partial left, middle, partial right
        # the same approach as `_range_min_select` but summing counts
        # partial left
        if Lb == Rb:
            return self._count_local_min(i, j+1, min_val)
        left_end = min((Lb+1)*b, self.size)
        cnt += self._count_local_min(i, left_end, min_val)

        if Rb - 1 >= Lb+1:
            # blocks [Lb+1..Rb-1]
            cnt += self._count_blocks_value(Lb+1, Rb-1, min_val, is_min=True)

        right_start = Rb*b
        if right_start <= j:
            cnt += self._count_local_min(right_start, j+1, min_val)
        return cnt

    def minselect(self, i, j, q):
        """
        position of the qth minimum in excess(i..j) in left-to-right order.
        We do basically the same approach as rmq but with partial-block merges.
        """
        if j < i:
            return None
        min_val = self._range_min_value(i, j)
        return self._range_min_select(i, j, min_val, q)

    def nsibling(self, i):
        if self.B[i]:
            pos = self.close(i) + 1
        else:
            pos = self.nsibling(self.open(i))
        if pos >= self.size:
            return 0
        elif self.B[pos]:
            return pos
        else:
            return 0

    def psibling(self, i):
        if self.B[i]:
            if i == 0:
                return 0
            if self.B[i-1]:
                return 0
            pos = self.open(i-1)
        else:
            pos = self.psibling(self.open(i))
        if pos < 0:
            return 0
        elif self.B[pos]:
            return pos
        else:
            return 0

    def preorder(self, i):
        if self.B[i]:
            return self.rank(1, i)
        else:
            return self.preorder(self.open(i))

    def preorderselect(self, k):
        return self.select(1, k)

    def postorder(self, i):
        if self.B[i]:
            return self.rank(0, self.close(i))
        else:
            return self.rank(0, i)

    def postorderselect(self, k):
        return self.open(self.select(0, k))

    def isancestor(self, i, j):
        if i == j:
            return False
        if not self.B[i]:
            i = self.open(i)
        return (i <= j < self.close(i))

    def subtree(self, i):
        if not self.B[i]:
            i = self.open(i)
        return (self.close(i) - i + 1) // 2

    def levelancestor(self, i, d):
        if d <= 0:
            return -1
        if not self.B[i]:
            i = self.open(i)
        return self.bwdsearch(i, -(d+1)) + 1

    def levelnext(self, i):
        return self.fwdsearch(self.close(i), 1)

    def lca(self, i, j):
        if self.isancestor(i, j):
            return i
        elif self.isancestor(j, i):
            return j
        else:
            return self.parent(self.rmq(i, j) + 1)

    def deepestnode(self, i):
        return self.rMq(self.open(i), self.close(i))

    def height(self, i):
        return self._excess(self.deepestnode(i)) - self._excess(self.open(i))

    def ntips(self):
        i = 0
        count = 0
        n = self.size
        while i < (n - 1):
            if self.B[i] and not self.B[i+1]:
                count += 1
                i += 1
            i += 1
        return count

    def shear(self, tips):
        n = len(self.B)
        mask = [False] * n

        # Mark the root
        mask[self.root()] = True
        mask[self.close(self.root())] = True

        count = 0
        for i in range(n):
            if self.isleaf(i):
                if self.name(i) in tips:
                    count += 1
                    mask[i] = True
                    if i+1 < n:
                        mask[i+1] = True
                    p = self.parent(i)
                    while p != 0 and (not mask[p]):
                        mask[p] = True
                        c = self.close(p)
                        if c < n:
                            mask[c] = True
                        p = self.parent(p)
        if count == 0:
            raise ValueError("No requested tips found")

        return self._mask_from_self(mask, self._lengths)

    def _mask_from_self(self, mask, lengths):
        indices = [i for i, mval in enumerate(mask) if mval]
        new_b = np.empty(len(indices), dtype=bool)
        new_names = np.empty(len(indices), dtype=object)
        new_lengths = np.empty(len(indices), dtype=float)

        for k, i in enumerate(indices):
            new_b[k] = self.B[i]
            new_names[k] = self._names[i]
            new_lengths[k] = lengths[i]

        return BP(new_b, names=new_names, lengths=new_lengths)

    def collapse(self):
        n = np.sum(self.B)
        mask = [False] * len(self.B)
        mask[self.root()] = True
        mask[self.close(self.root())] = True

        new_lengths = self._lengths.copy()

        for i in range(n):
            current = self.preorderselect(i)
            if self.isleaf(current):
                mask[current] = True
                c = self.close(current)
                if c < len(self.B):
                    mask[c] = True
            else:
                first = self.fchild(current)
                last = self.lchild(current)
                if (first == last) and (first != 0):
                    new_lengths[first] += new_lengths[current]
                else:
                    mask[current] = True
                    c = self.close(current)
                    if c < len(self.B):
                        mask[c] = True

        return self._mask_from_self(mask, new_lengths)
    