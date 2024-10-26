class BP_Tree:
    def __init__(self, bitstring):
        """
        Initializes the BP_Tree with a given bitstring.

        Parameters:
        - bitstring (str): A string consisting of '0's and '1's representing the BP sequence.
        """
        self.bitvector = BitVector(bitstring)
        self.n = self.bitvector.n
        self.rmm_tree = RMMTree(self.bitvector)

    def excess(self, i):
        """
        Returns the excess at position i.

        Parameters:
        - i (int): The index at which to compute excess.

        Returns:
        - int: The excess value at position i.

        Raises:
        - IndexError: If i is out of bounds.
        """
        if i < 0 or i >= self.n:
            raise IndexError("Index out of bounds.")
        return 2 * self.bitvector.rank1(i) - (i + 1)

    def find_close(self, i):
        """
        Finds the matching closing parenthesis for the opening parenthesis at position i.

        Parameters:
        - i (int): Index of the opening parenthesis.

        Returns:
        - int: Index of the matching closing parenthesis.

        Raises:
        - IndexError: If i is out of bounds.
        - ValueError: If there's no opening parenthesis at position i or no matching closing parenthesis.
        """
        if i < 0 or i >= self.n:
            raise IndexError("Index out of bounds.")
        if not self.bitvector.B[i]:
            raise ValueError(f"No opening parenthesis at position {i}.")

        target_excess = self.excess(i) - 1
        # Find the position with the minimum excess in the range [i + 1, n - 1]
        min_pos = self.rmm_tree.query_min(i + 1, self.n - 1)

        if self.excess(min_pos) == target_excess:
            return min_pos
        else:
            raise ValueError("Matching closing parenthesis not found.")

    def find_open(self, i):
        """
        Finds the matching opening parenthesis for the closing parenthesis at position i.

        Parameters:
        - i (int): Index of the closing parenthesis.

        Returns:
        - int: Index of the matching opening parenthesis.

        Raises:
        - IndexError: If i is out of bounds.
        - ValueError: If there's no closing parenthesis at position i or no matching opening parenthesis.
        """
        if i < 0 or i >= self.n:
            raise IndexError("Index out of bounds.")
        if self.bitvector.B[i]:
            raise ValueError(f"No closing parenthesis at position {i}.")

        target_excess = self.excess(i) + 1
        # Find the position with the minimum excess in the range [0, i - 1]
        min_pos = self.rmm_tree.query_min(0, i - 1)

        if self.excess(min_pos) == target_excess:
            return min_pos
        else:
            raise ValueError("Matching opening parenthesis not found.")

    def lca(self, i, j):
        """
        Computes the Lowest Common Ancestor (LCA) of nodes at positions i and j.

        Parameters:
        - i (int): Index of the first node.
        - j (int): Index of the second node.

        Returns:
        - int: Index of the LCA node.

        Raises:
        - IndexError: If i or j is out of bounds.
        """
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            raise IndexError("Index out of bounds.")
        if i > j:
            i, j = j, i  # Ensure i <= j

        # Find the position with the minimum excess in the range [i, j]
        min_pos = self.rmm_tree.query_min(i, j)
        return self.find_open(min_pos)

    def depth(self, i):
        """
        Computes the depth of the node at position i.

        Parameters:
        - i (int): Index of the node.

        Returns:
        - int: The depth of the node.

        Raises:
        - IndexError: If i is out of bounds.
        """
        if i < 0 or i >= self.n:
            raise IndexError("Index out of bounds.")
        return (self.excess(i) + 1) // 2

    def parent(self, i):
        """
        Finds the parent of the node at position i.

        Parameters:
        - i (int): Index of the node.

        Returns:
        - int or None: Index of the parent node or None if it's the root.

        Raises:
        - IndexError: If i is out of bounds.
        """
        if i <= 0 or i >= self.n:
            return None  # Root node or invalid index has no parent
        return self.enclose(i)

    def enclose(self, i):
        """
        Finds the position of the opening parenthesis that encloses the one at position i.

        Parameters:
        - i (int): Index of the node.

        Returns:
        - int or None: Index of the enclosing parenthesis or None if it's the root.

        Raises:
        - IndexError: If i is out of bounds.
        """
        if i <= 0 or i >= self.n:
            raise IndexError("Index out of bounds.")
        d = self.excess(i) - 1
        # Find the position with the minimum excess in the range [0, i - 1]
        min_pos = self.rmm_tree.query_min(0, i - 1)
        if self.excess(min_pos) == d:
            return min_pos
        else:
            return None
