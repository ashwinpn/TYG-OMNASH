class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Count the frequency of each number using Counter
        # Complexity: O(n), where n is the length of nums
        freq_pairs = [(num,freq) for num,freq in Counter(nums).items()]

        # Sort the frequency pairs by frequency in descending order
        # Complexity: O(n log n) for sorting
        freq_pairs.sort(key = lambda x : x[1], reverse = True)

        # Return the top k elements
        # Complexity: O(k)
        return [num for (num,freq) in freq_pairs[:k]]

# Overall Complexity: O(n log n)

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # Initialize a heap
        # Complexity: O(1)
        heap = []

        for x, y in points:
            # Calculate negative distance to use max-heap as min-heap
            # Complexity: O(1)
            dist = -(x*x + y*y)

            if len(heap) >= k:
                # Push new point and pop the farthest point
                # Complexity: O(log k)
                heapq.heappushpop(heap, (dist, x, y))
            else:
                # Push point to heap
                # Complexity: O(log k)
                heapq.heappush(heap, (dist, x, y))

        # Extract k closest points
        # Complexity: O(k)
        return [[x, y] for (dist, x, y) in heap]

# Overall Complexity: O(n log k)

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # Create a min-heap of the first k elements
        # Complexity: O(k)
        heap = nums[:k]

        # Heapify the k elements
        # Complexity: O(k)
        heapq.heapify(heap)

        for n in nums[k:]:
            if n > heap[0]:
                # Push new element and pop the smallest element
                # Complexity: O(log k)
                heapq.heappushpop(heap, n)

        # Return the root of the heap
        # Complexity: O(1)
        return heap[0]

# Overall Complexity: O(n log k)

class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Dictionary to hold column-indexed nodes
        # Complexity: O(1)
        ct = defaultdict(list)

        def dfs(node, row, col):
            if node is not None:
                # Add the node's value to the corresponding column
                # Complexity: O(1)
                ct[col].append((row, node.val))

                # Recursively visit left and right subtrees
                # Complexity: O(n) for all nodes
                dfs(node.left, row + 1, col - 1)
                dfs(node.right, row + 1, col + 1)

        # Start DFS traversal
        # Complexity: O(n)
        dfs(root, 0, 0)

        # Sort columns and rows within each column
        # Complexity: O(n log n) for sorting
        return [[val for (row, val) in sorted(ct[column])] for column in sorted(ct)]

# Overall Complexity: O(n log n)

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # Get dimensions of the matrix
        # Complexity: O(1)
        rows, cols = len(matrix), len(matrix[0])

        # Transpose the matrix
        # Complexity: O(n^2), where n is the dimension of the matrix
        for r in range(rows):
            for c in range(r, cols):
                matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]

        # Reverse each row
        # Complexity: O(n^2)
        for row in matrix:
            row.reverse()

# Overall Complexity: O(n^2)

class Solution:
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        # Initialize graph and visited array
        # Complexity: O(n)
        graph = defaultdict(list)
        visited = [False] * n
        num_completed = 0

        # Build adjacency list
        # Complexity: O(e), where e is the number of edges
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)

        def dfs(node):
            nv, ne = 1, len(graph[node])
            visited[node] = True

            for neb in graph[node]:
                if not visited[neb]:
                    v, e = dfs(neb)
                    nv += v
                    ne += e

            return nv, ne

        # Traverse all nodes
        # Complexity: O(n + e)
        for i in range(n):
            if not visited[i]:
                v, e = dfs(i)
                if e == v * (v - 1):
                    num_completed += 1

        return num_completed

# Overall Complexity: O(n + e)

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # Initialize variables
        # Complexity: O(1)
        num_islands = 0
        rows, cols = len(grid), len(grid[0])

        def dfs(r, c):
            if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] == "0":
                return

            # Mark the cell as visited
            # Complexity: O(1)
            grid[r][c] = "0"

            # Explore all directions
            # Complexity: O(4) per cell
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                dfs(r + dr, c + dc)

        # Traverse the grid
        # Complexity: O(m * n), where m is rows and n is columns
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == "1":
                    num_islands += 1
                    dfs(i, j)

        return num_islands

# Overall Complexity: O(m * n)

class TimeMap:
    def __init__(self):
        # Initialize storage dictionary
        # Complexity: O(1)
        self.store = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        # Append the key-value pair with timestamp
        # Complexity: O(1)
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        # Binary search for the closest timestamp
        # Complexity: O(log k), where k is the number of timestamps for the key
        values = self.store.get(key, [])
        i = bisect_right(values, (timestamp, chr(127))) - 1
        return values[i][1] if i >= 0 else ""

# Overall Complexity for set: O(1), get: O(log k)

class LRUCache:
    def __init__(self, capacity: int):
        # Initialize cache and capacity
        # Complexity: O(1)
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        # Retrieve value if key exists and move key to end
        # Complexity: O(1)
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        # Update or insert the key-value pair
        # Complexity: O(1)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        # Remove the least recently used item if over capacity
        # Complexity: O(1)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Overall Complexity for get and put: O(1)
