"""
Fibonacci Heap Implementation
Advanced priority queue with O(1) amortized insert and decrease-key operations
"""

import math
from typing import Optional, Any, Dict, List

class FibNode:
    """Node in a Fibonacci Heap"""
    def __init__(self, key: Any, data: Any = None):
        self.key = key
        self.data = data
        self.parent: Optional['FibNode'] = None
        self.child: Optional['FibNode'] = None
        self.left: Optional['FibNode'] = self
        self.right: Optional['FibNode'] = self
        self.degree = 0
        self.mark = False
    
    def __lt__(self, other):
        return self.key < other.key
    
    def __repr__(self):
        return f"FibNode(key={self.key}, data={self.data})"

class FibonacciHeap:
    """
    Fibonacci Heap implementation optimized for Huffman tree construction
    
    Key Operations:
    - insert: O(1) amortized
    - extract_min: O(log n) amortized  
    - decrease_key: O(1) amortized
    - merge: O(1)
    """
    
    def __init__(self):
        self.min_node: Optional[FibNode] = None
        self.num_nodes = 0
        self.node_map: Dict[Any, FibNode] = {}  # For decrease_key operations
    
    def is_empty(self) -> bool:
        """Check if heap is empty"""
        return self.num_nodes == 0
    
    def insert(self, key: Any, data: Any = None) -> FibNode:
        """Insert new node with given key and data"""
        node = FibNode(key, data)
        
        # Store reference for potential decrease_key operations
        self.node_map[id(node)] = node
        
        if self.min_node is None:
            self.min_node = node
        else:
            # Add to root list
            self._add_to_root_list(node)
            if node.key < self.min_node.key:
                self.min_node = node
        
        self.num_nodes += 1
        return node
    
    def get_min(self) -> Optional[FibNode]:
        """Get minimum node without removing it"""
        return self.min_node
    
    def extract_min(self) -> Optional[FibNode]:
        """Extract and return minimum node"""
        min_node = self.min_node
        
        if min_node is None:
            return None
        
        # Move children to root list
        if min_node.child:
            children = self._get_node_list(min_node.child)
            for child in children:
                child.parent = None
                self._add_to_root_list(child)
        
        # Remove min_node from root list
        self._remove_from_root_list(min_node)
        
        if min_node == min_node.right:
            # Was the only node
            self.min_node = None
        else:
            self.min_node = min_node.right
            self._consolidate()
        
        self.num_nodes -= 1
        
        # Clean up node_map
        if id(min_node) in self.node_map:
            del self.node_map[id(min_node)]
        
        return min_node
    
    def decrease_key(self, node: FibNode, new_key: Any):
        """Decrease key of given node"""
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        
        node.key = new_key
        parent = node.parent
        
        if parent and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        
        if node.key < self.min_node.key:
            self.min_node = node
    
    def merge(self, other: 'FibonacciHeap') -> 'FibonacciHeap':
        """Merge two Fibonacci heaps"""
        if other.is_empty():
            return self
        
        if self.is_empty():
            self.min_node = other.min_node
            self.num_nodes = other.num_nodes
            self.node_map.update(other.node_map)
            return self
        
        # Merge root lists
        self._merge_root_lists(other.min_node)
        
        if other.min_node.key < self.min_node.key:
            self.min_node = other.min_node
        
        self.num_nodes += other.num_nodes
        self.node_map.update(other.node_map)
        
        return self
    
    def _add_to_root_list(self, node: FibNode):
        """Add node to root list"""
        if self.min_node is None:
            self.min_node = node
            node.left = node.right = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node
    
    def _remove_from_root_list(self, node: FibNode):
        """Remove node from root list"""
        if node.right == node:
            return
        node.left.right = node.right
        node.right.left = node.left
    
    def _consolidate(self):
        """Consolidate trees of same degree"""
        max_degree = int(math.log(self.num_nodes, 2)) + 1
        degree_table: List[Optional[FibNode]] = [None] * (max_degree + 1)
        
        # Get all root nodes
        roots = self._get_root_list()
        
        for node in roots:
            degree = node.degree
            
            while degree_table[degree] is not None:
                other = degree_table[degree]
                
                if node.key > other.key:
                    node, other = other, node
                
                # Make other a child of node
                self._link(other, node)
                degree_table[degree] = None
                degree += 1
            
            degree_table[degree] = node
        
        # Rebuild root list and find new minimum
        self.min_node = None
        for node in degree_table:
            if node is not None:
                if self.min_node is None:
                    self.min_node = node
                    node.left = node.right = node
                else:
                    self._add_to_root_list(node)
                    if node.key < self.min_node.key:
                        self.min_node = node
    
    def _link(self, child: FibNode, parent: FibNode):
        """Make child a child of parent"""
        # Remove child from root list
        self._remove_from_root_list(child)
        
        # Make child a child of parent
        if parent.child is None:
            parent.child = child
            child.left = child.right = child
        else:
            child.left = parent.child
            child.right = parent.child.right
            parent.child.right.left = child
            parent.child.right = child
        
        child.parent = parent
        parent.degree += 1
        child.mark = False
    
    def _cut(self, node: FibNode, parent: FibNode):
        """Cut node from parent and add to root list"""
        # Remove node from parent's child list
        if parent.child == node:
            if node.right == node:
                parent.child = None
            else:
                parent.child = node.right
        
        node.left.right = node.right
        node.right.left = node.left
        parent.degree -= 1
        
        # Add to root list
        node.parent = None
        node.mark = False
        self._add_to_root_list(node)
    
    def _cascading_cut(self, node: FibNode):
        """Perform cascading cut"""
        parent = node.parent
        
        if parent is not None:
            if not node.mark:
                node.mark = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)
    
    def _get_root_list(self) -> List[FibNode]:
        """Get list of all root nodes"""
        if self.min_node is None:
            return []
        
        roots = []
        current = self.min_node
        
        while True:
            roots.append(current)
            current = current.right
            if current == self.min_node:
                break
        
        return roots
    
    def _get_node_list(self, start_node: FibNode) -> List[FibNode]:
        """Get circular list of nodes starting from given node"""
        nodes = []
        current = start_node
        
        while True:
            nodes.append(current)
            current = current.right
            if current == start_node:
                break
        
        return nodes
    
    def _merge_root_lists(self, other_min: FibNode):
        """Merge root list with another heap's root list"""
        if self.min_node is None or other_min is None:
            return
        
        # Connect the two circular lists
        self.min_node.right.left = other_min.left
        other_min.left.right = self.min_node.right
        self.min_node.right = other_min
        other_min.left = self.min_node
    
    def size(self) -> int:
        """Return number of nodes in heap"""
        return self.num_nodes
    
    def __len__(self) -> int:
        return self.num_nodes
    
    def __bool__(self) -> bool:
        return not self.is_empty()
    
    def display_heap(self):
        """Display heap structure (for debugging)"""
        if self.is_empty():
            print("Empty heap")
            return
        
        print(f"Heap with {self.num_nodes} nodes, min = {self.min_node.key}")
        roots = self._get_root_list()
        for i, root in enumerate(roots):
            print(f"Root {i}: {root.key} (degree: {root.degree})")
