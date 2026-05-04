from typing import List

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        empty_cells = []
        
        # Initialize sets and collect empty cells
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    empty_cells.append((i, j))
                else:
                    num = board[i][j]
                    rows[i].add(num)
                    cols[j].add(num)
                    box_idx = (i // 3) * 3 + (j // 3)
                    boxes[box_idx].add(num)
        
        def get_candidates(i: int, j: int) -> List[str]:
            """Return all valid candidates for a cell"""
            box_idx = (i // 3) * 3 + (j // 3)
            used = rows[i] | cols[j] | boxes[box_idx]
            return [str(num) for num in range(1, 10) if str(num) not in used]
        
        def backtrack(index: int) -> bool:
            if index == len(empty_cells):
                return True
            
            # Find the cell with fewest candidates (optimization)
            min_candidates = 10
            min_pos = -1
            
            for pos in range(index, len(empty_cells)):
                i, j = empty_cells[pos]
                candidates = get_candidates(i, j)
                if len(candidates) < min_candidates:
                    min_candidates = len(candidates)
                    min_pos = pos
                    if min_candidates == 1:
                        break
            
            # Swap to put the most constrained cell first
            if min_pos != index:
                empty_cells[index], empty_cells[min_pos] = empty_cells[min_pos], empty_cells[index]
            
            i, j = empty_cells[index]
            candidates = get_candidates(i, j)
            box_idx = (i // 3) * 3 + (j // 3)
            
            for num in candidates:
                # Place the number
                board[i][j] = num
                rows[i].add(num)
                cols[j].add(num)
                boxes[box_idx].add(num)
                
                if backtrack(index + 1):
                    return True
                
                # Undo
                board[i][j] = '.'
                rows[i].remove(num)
                cols[j].remove(num)
                boxes[box_idx].remove(num)
            
            # Swap back
            if min_pos != index:
                empty_cells[index], empty_cells[min_pos] = empty_cells[min_pos], empty_cells[index]
            
            return False
        
        backtrack(0)
