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
            b = int(math.sqrt(self.n)) or 1
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
        scan_end = min(i, self.n)
        
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

    # For debugging
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


# ------------------------------------------------------------------
# TESTING
# ------------------------------------------------------------------

def test_empty_parentheses():
    print("\n--- Test: Empty String ---")
    rmm = RMMTree("")
    root_summary = rmm.get_root_summary()
    print("Root summary for empty:", root_summary)
    assert root_summary is None, "Empty string should yield None root summary"
    print("PASS: Empty string")

def test_balanced_small():
    print("\n--- Test: Balanced '()' ---")
    rmm = RMMTree("()")
    root_summary = rmm.get_root_summary()
    print("Root summary for '()': ", root_summary)
    assert root_summary == (0, 0, 1, 1), "Expected (0, 0, 1, 1)"
    # Test get_excess
    assert rmm.get_excess(0) == 0, "prefix_excess[0] should be 0"
    assert rmm.get_excess(1) == 1, "prefix_excess[1] should be 1"
    assert rmm.get_excess(2) == 0, "prefix_excess[2] should be 0"
    print("PASS: Balanced '()'")

def test_single_parenthesis():
    print("\n--- Test: Single '(' ---")
    rmm = RMMTree("(")
    root_summary = rmm.get_root_summary()
    print("Root summary for '(': ", root_summary)
    assert root_summary == (1, 1, 1, 1), "Expected (1, 1, 1, 1)"
    # Test get_excess
    assert rmm.get_excess(0) == 0, "prefix_excess[0] should be 0"
    assert rmm.get_excess(1) == 1, "prefix_excess[1] should be 1"
    print("PASS: Single '('")

def test_unbalanced_sequence():
    print("\n--- Test: Unbalanced Sequence ')(' ---")
    parentheses = ")("  
    rmm = RMMTree(parentheses, b=2)
    
    root_summary = rmm.get_root_summary()
    print("Root summary for ')(': ", root_summary)
    # Expected: (0, -1, 0, 1)
    
    assert root_summary == (0, -1, 0, 1), "Expected root summary (0, -1, 0, 1)"
    print("PASS: Unbalanced sequence ')('")

def test_sample_construction():
    print("\n--- Test: Sample Construction + Queries ---")
    parentheses = "(((())(()))()((()))())"
    rmm = RMMTree(parentheses, b=4)
    
    root = rmm.get_root_summary()
    print("Root summary:", root)
    net_excess = 0
    for c in parentheses:
        if c == '(':
            net_excess += 1
        else:
            net_excess -= 1
    print(f"Net excess (should be 0): {net_excess}")
    assert net_excess == 0, "Sample parentheses should be balanced => net_excess=0"
    assert root[0] == 0, "Root summary e should be 0 for balanced parentheses."
    
    min_val = rmm.range_min_query(1, 3)
    max_val = rmm.range_max_query(1, 3)
    print(f"Block-range [1..3] => min prefix-excess = {min_val}, max prefix-excess = {max_val}")
    
    summary_10_17 = rmm.query_range_characters(10, 17)
    print(f"Character-range [10..17] => summary = {summary_10_17}")
    print("PASS: Sample construction test.")

def test_get_excess():
    print("\n--- Test: get_excess ---")
    parentheses = "((()()(()))()((()())))"  # Example sequence
    rmm = RMMTree(parentheses, b=4)
    
    # Manually compute prefix_excess
    prefix = [0]
    current = 0
    for c in parentheses:
        if c == '(':
            current += 1
        else:
            current -= 1
        prefix.append(current)
    
    # Test get_excess for all positions
    for i in range(len(prefix)):
        computed = rmm.get_excess(i)
        expected = prefix[i]
        print(f"get_excess({i}) = {computed}, expected = {expected}")
        assert computed == expected, f"prefix_excess[{i}] should be {expected}, got {computed}"
    
    print("PASS: get_excess tests.")

def run_all_tests():
    test_empty_parentheses()
    test_single_parenthesis()
    test_balanced_small()
    test_sample_construction()
    test_get_excess()
    test_unbalanced_sequence() 
    print("\nAll tests passed!")

if __name__ == "__main__":
    run_all_tests()
    
    # Printing the example tree from figure 2
    parentheses_sequence = "((()()(()))()((()())))" 
    print('\n')
    print('Example Sequence:', parentheses_sequence)
    rmm = RMMTree(parentheses_sequence, b=4)  
    rmm.print_tree() 
    

# TODO: Error handling, more testing, additional functions
