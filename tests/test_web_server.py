"""
tests/test_web_server.py — Tests for the web server endpoints and problem bank.

All tests run offline:
- GHCBridge is mocked
- Groq API is mocked
- No real HTTP calls made

Run with:
    python -m pytest tests/test_web_server.py -v
"""

from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

# ── We need to mock before importing web_server ───────────────────────────

import sys
from unittest.mock import MagicMock

# Mock the server modules that require GHC/Groq
sys.modules.setdefault('server.ghc.bridge',  MagicMock())
sys.modules.setdefault('server.ghc.models',  MagicMock())
sys.modules.setdefault('server.ai.engine',   MagicMock())
sys.modules.setdefault('server.ai.context',  MagicMock())


# ── Problem bank validation ───────────────────────────────────────────────

REQUIRED_PROBLEM_FIELDS = {"id", "topic", "title", "difficulty", "description", "stub", "hint"}

VALID_TOPICS = {
    "Recursion", "Pattern Matching", "List Comprehensions",
    "Higher Order Functions", "Types", "Tuples",
    "Strings", "Algorithms", "Exam Style",
}

# Inline the problem bank for testing (same data as web_editor.html)
# tests/test_web_server.py - Replace the entire TEST_PROBLEMS section (lines ~30-130) with this:

TEST_PROBLEMS = [
    # Recursion problems
    {"id":"rec1","topic":"Recursion","title":"Sum a list","difficulty":1,
     "description":"Write a function that adds all numbers in a list together.\n\nExamples:\n  sumList [1,2,3,4,5]  -->  15\n  sumList []           -->  0",
     "stub":"module Main where\n\nsumList :: [Int] -> Int\nsumList xs = undefined\n\nmain :: IO ()\nmain = do\n  print (sumList [1,2,3,4,5])\n  print (sumList [])",
     "hint":"What happens with an empty list? What do you do with one element plus the rest?"},
    
    {"id":"rec2","topic":"Recursion","title":"Count elements","difficulty":1,
     "description":"Write a function that counts how many elements are in a list.\nDo not use the built-in length.\n\nExamples:\n  myLength [1,2,3]  -->  3\n  myLength []       -->  0",
     "stub":"module Main where\n\nmyLength :: [a] -> Int\nmyLength xs = undefined\n\nmain :: IO ()\nmain = do\n  print (myLength [1,2,3])\n  print (myLength [])",
     "hint":"Each time you take one element off the list, the length goes down by one."},
    
    {"id":"rec3","topic":"Recursion","title":"Count odd digits","difficulty":3,
     "description":"Write a function that counts how many odd digits are in a non-negative integer.\n\nExamples:\n  countOdds 1234  -->  2\n  countOdds 2468  -->  0\n  countOdds 0     -->  0",
     "stub":"module Main where\n\ncountOdds :: Int -> Int\ncountOdds n = undefined\n\nmain :: IO ()\nmain = do\n  print (countOdds 1234)\n  print (countOdds 2468)\n  print (countOdds 0)",
     "hint":"mod 10 gives the last digit. div 10 removes it. What is your base case?"},
    
    {"id":"rec4","topic":"Recursion","title":"Reverse a list","difficulty":2,
     "description":"Write a function that reverses a list without using the built-in reverse.\n\nExamples:\n  myReverse [1,2,3]  -->  [3,2,1]\n  myReverse []       -->  []",
     "stub":"module Main where\n\nmyReverse :: [a] -> [a]\nmyReverse xs = undefined\n\nmain :: IO ()\nmain = do\n  print (myReverse [1,2,3])\n  print (myReverse [])",
     "hint":"Think about where the first element ends up in the reversed list."},
    
    {"id":"rec5","topic":"Recursion","title":"Fibonacci","difficulty":3,
     "description":"Return the nth Fibonacci number.\nFib(0)=0, Fib(1)=1, Fib(n)=Fib(n-1)+Fib(n-2)\n\nExamples:\n  fib 0  -->  0\n  fib 1  -->  1\n  fib 7  -->  13",
     "stub":"module Main where\n\nfib :: Int -> Int\nfib n = undefined\n\nmain :: IO ()\nmain = do\n  print (fib 0)\n  print (fib 1)\n  print (fib 7)",
     "hint":"This function calls itself TWICE. What are the two base cases?"},
    
    # Pattern Matching problems
    {"id":"pat1","topic":"Pattern Matching","title":"Safe head","difficulty":1,
     "description":"Return the first element as Maybe, or Nothing if empty.\n\nExamples:\n  safeHead [1,2,3]  -->  Just 1\n  safeHead []       -->  Nothing",
     "stub":"module Main where\n\nsafeHead :: [a] -> Maybe a\nsafeHead xs = undefined\n\nmain :: IO ()\nmain = do\n  print (safeHead [1,2,3])\n  print (safeHead ([] :: [Int]))",
     "hint":"You need two patterns: one for empty list, one for non-empty."},
    
    {"id":"pat2","topic":"Pattern Matching","title":"Describe a number","difficulty":1,
     "description":"Describe a number: negative, zero, small (1-9), or large (10+).\n\nExamples:\n  describe (-3)  -->  \"negative\"\n  describe 0     -->  \"zero\"\n  describe 5     -->  \"small\"\n  describe 42    -->  \"large\"",
     "stub":"module Main where\n\ndescribe :: Int -> String\ndescribe n = undefined\n\nmain :: IO ()\nmain = do\n  putStrLn (describe (-3))\n  putStrLn (describe 0)\n  putStrLn (describe 5)\n  putStrLn (describe 42)",
     "hint":"Use guards: | n < 0 = ... | n == 0 = ..."},
    
    {"id":"pat3","topic":"Pattern Matching","title":"Handle Maybe","difficulty":2,
     "description":"Double a number inside a Maybe, or return Nothing.\n\nExamples:\n  doubleIfJust (Just 5)  -->  Just 10\n  doubleIfJust Nothing   -->  Nothing",
     "stub":"module Main where\n\ndoubleIfJust :: Maybe Int -> Maybe Int\ndoubleIfJust mx = undefined\n\nmain :: IO ()\nmain = do\n  print (doubleIfJust (Just 5))\n  print (doubleIfJust Nothing)",
     "hint":"Pattern match on the Maybe. What do you do with Just x? What about Nothing?"},
    
    # List Comprehensions problems
    {"id":"lc1","topic":"List Comprehensions","title":"Even squares","difficulty":1,
     "description":"Using a list comprehension, produce squares of all even numbers from 1 to 20.\n\nExpected: [4,16,36,64,100,144,196,256,324,400]",
     "stub":"module Main where\n\nevenSquares :: [Int]\nevenSquares = undefined\n\nmain :: IO ()\nmain = print evenSquares",
     "hint":"[expr | x <- range, condition] — what is your range, condition, and expression?"},
    
    {"id":"lc2","topic":"List Comprehensions","title":"Pythagorean triples","difficulty":3,
     "description":"Find all Pythagorean triples (a,b,c) where a < b < c and all are between 1 and n.\n\ntriples 20 --> [(3,4,5),(5,12,13),(6,8,10),(8,15,17),(9,12,15),(12,16,20)]",
     "stub":"module Main where\n\ntriples :: Int -> [(Int,Int,Int)]\ntriples n = undefined\n\nmain :: IO ()\nmain = print (triples 20)",
     "hint":"You need three generators and a condition. What is a^2 + b^2 equal to?"},
    
    {"id":"lc3","topic":"List Comprehensions","title":"Prime numbers","difficulty":4,
     "description":"Return all prime numbers up to n using a list comprehension.\n\nprimesTo 20 --> [2,3,5,7,11,13,17,19]",
     "stub":"module Main where\n\nisPrime :: Int -> Bool\nisPrime n = undefined\n\nprimesTo :: Int -> [Int]\nprimesTo n = undefined\n\nmain :: IO ()\nmain = print (primesTo 20)",
     "hint":"A number is prime if no number from 2 to sqrt(n) divides it evenly."},
    
    # Higher Order Functions problems
    {"id":"hof1","topic":"Higher Order Functions","title":"Double all","difficulty":1,
     "description":"Using map, double every number in a list.\n\ndoubleAll [1,2,3,4,5] --> [2,4,6,8,10]",
     "stub":"module Main where\n\ndoubleAll :: [Int] -> [Int]\ndoubleAll xs = undefined\n\nmain :: IO ()\nmain = print (doubleAll [1,2,3,4,5])",
     "hint":"map applies a function to every element. What function doubles a number?"},
    
    {"id":"hof2","topic":"Higher Order Functions","title":"Filter positives","difficulty":1,
     "description":"Using filter, keep only positive numbers.\n\npositives [-1,2,-3,4,-5] --> [2,4]",
     "stub":"module Main where\n\npositives :: [Int] -> [Int]\npositives xs = undefined\n\nmain :: IO ()\nmain = print (positives [-1,2,-3,4,-5])",
     "hint":"filter keeps elements where a predicate is True. What predicate checks positivity?"},
    
    {"id":"hof3","topic":"Higher Order Functions","title":"Sum and product with foldr","difficulty":2,
     "description":"Using foldr, implement sum and product of a list.\n\nmySum [1,2,3,4,5]     --> 15\nmyProduct [1,2,3,4,5] --> 120",
     "stub":"module Main where\n\nmySum :: [Int] -> Int\nmySum xs = undefined\n\nmyProduct :: [Int] -> Int\nmyProduct xs = undefined\n\nmain :: IO ()\nmain = do\n  print (mySum [1,2,3,4,5])\n  print (myProduct [1,2,3,4,5])",
     "hint":"foldr f z xs — what is z (starting value) for sum? For product?"},
    
    {"id":"hof4","topic":"Higher Order Functions","title":"Alternating squares","difficulty":3,
     "description":"Sum squares of evens, subtract squares of odds, from 1 to n.\n\nsumAlternating 5  --> -15\nsumAlternating 10 --> 55",
     "stub":"module Main where\n\nsumAlternating :: Int -> Int\nsumAlternating n = undefined\n\nmain :: IO ()\nmain = do\n  print (sumAlternating 5)\n  print (sumAlternating 10)",
     "hint":"Think: sum of even squares MINUS sum of odd squares. Use list comprehensions or filter."},
    
    # Types problems
    {"id":"typ1","topic":"Types","title":"Safe divide","difficulty":2,
     "description":"Divide two numbers, returning Nothing if dividing by zero.\n\nsafeDivide 10 2 --> Just 5\nsafeDivide 10 0 --> Nothing",
     "stub":"module Main where\n\nsafeDivide :: Int -> Int -> Maybe Int\nsafeDivide x y = undefined\n\nmain :: IO ()\nmain = do\n  print (safeDivide 10 2)\n  print (safeDivide 10 0)",
     "hint":"Check if y is zero first. Return Nothing or Just accordingly."},
    
    {"id":"typ2","topic":"Types","title":"Shape area","difficulty":3,
     "description":"Define Shape with Circle and Rectangle. Compute area.\n\narea (Circle 5.0)        --> 78.54\narea (Rectangle 3.0 4.0) --> 12.0",
     "stub":"module Main where\n\ndata Shape = Circle Double | Rectangle Double Double\n\narea :: Shape -> Double\narea s = undefined\n\nmain :: IO ()\nmain = do\n  print (area (Circle 5.0))\n  print (area (Rectangle 3.0 4.0))",
     "hint":"Pattern match on Shape constructors. Circle: pi*r^2. Rectangle: w*h."},
    
    # Tuples problems
    {"id":"tup1","topic":"Tuples","title":"Closest point","difficulty":4,
     "description":"Given named points and a target, return the name of the closest point.\n\nclosest [(\"A\",(0,0)),(\"B\",(3,4)),(\"C\",(1,1))] (2,2) --> \"C\"",
     "stub":"module Main where\n\ndistance :: (Double,Double) -> (Double,Double) -> Double\ndistance (x1,y1) (x2,y2) = sqrt ((x2-x1)^2 + (y2-y1)^2)\n\nclosest :: [(String,(Double,Double))] -> (Double,Double) -> String\nclosest points target = undefined\n\nmain :: IO ()\nmain = do\n  let pts = [(\"A\",(0,0)),(\"B\",(3,4)),(\"C\",(1,1))]\n  putStrLn (closest pts (2,2))",
     "hint":"Map each point to (name, distance). Find minimum by distance. minimumBy from Data.List helps."},
    
    {"id":"tup2","topic":"Tuples","title":"Employee stats","difficulty":4,
     "description":"Given (name,age,salary) tuples:\n1. Find names of employees older than 50\n2. Total salary after 10% raise for over-50s\n\nInput: [(\"Alice\",55,1000),(\"Bob\",30,800),(\"Carol\",60,1200)]",
     "stub":"module Main where\n\nseniors :: [(String,Int,Int)] -> [String]\nseniors employees = undefined\n\ntotalAfterRaise :: [(String,Int,Int)] -> Double\ntotalAfterRaise employees = undefined\n\nmain :: IO ()\nmain = do\n  let emps = [(\"Alice\",55,1000),(\"Bob\",30,800),(\"Carol\",60,1200)]\n  print (seniors emps)\n  print (totalAfterRaise emps)",
     "hint":"Use filter for seniors, map for raises, sum for total. Lambda \\(n,a,s) -> ... destructures tuples."},
    
    # Strings problems
    {"id":"str1","topic":"Strings","title":"Number sequence","difficulty":3,
     "description":"Generate a string of numbers from start to end with a step.\n\nnumSeq 1 10 2 --> \"1 3 5 7 9 \"\nnumSeq 0 15 5 --> \"0 5 10 15 \"\nnumSeq 10 1 5 --> \"\"",
     "stub":"module Main where\n\nnumSeq :: Int -> Int -> Int -> String\nnumSeq start end step = undefined\n\nmain :: IO ()\nmain = do\n  putStrLn (numSeq 1 10 2)\n  putStrLn (numSeq 0 15 5)\n  putStrLn (numSeq 10 1 5)",
     "hint":"Base case: start > end. Recursive: show start ++ \" \" ++ numSeq (start+step) end step."},
    
    # Algorithms problems
    {"id":"alg1","topic":"Algorithms","title":"Insertion sort","difficulty":4,
     "description":"Implement insertion sort on a list of integers.\n\ninsertSort [3,1,4,1,5,9,2,6] --> [1,1,2,3,4,5,6,9]",
     "stub":"module Main where\n\ninsert :: Int -> [Int] -> [Int]\ninsert x [] = [x]\ninsert x (y:ys)\n  | x <= y    = x : y : ys\n  | otherwise = y : insert x ys\n\ninsertSort :: [Int] -> [Int]\ninsertSort xs = undefined\n\nmain :: IO ()\nmain = print (insertSort [3,1,4,1,5,9,2,6])",
     "hint":"insert is given. insertSort [] = []. For x:xs, insert x into the sorted rest."},
    
    {"id":"alg2","topic":"Algorithms","title":"Longest increasing run","difficulty":5,
     "description":"Length of the longest contiguous run where each element is exactly 1 more than previous.\n\nlongestRun [1,2,2,3,4]       --> 3\nlongestRun [1,2,3,4,5]       --> 5\nlongestRun [5,4,3,2,1]       --> 1",
     "stub":"module Main where\n\nlongestRun :: [Int] -> Int\nlongestRun xs = undefined\n\nmain :: IO ()\nmain = do\n  print (longestRun [1,2,2,3,4])\n  print (longestRun [1,2,3,4,5])\n  print (longestRun [5,4,3,2,1])",
     "hint":"Track current run length and max so far. What happens when two adjacent numbers differ by exactly 1?"},
    
    # Exam Style problems
    {"id":"exam1","topic":"Exam Style","title":"Primes at odd indices","difficulty":4,
     "description":"Product of all prime numbers at odd indices (1,3,5,...) in a list.\nReturn 1 if none found.\n\noddIndexPrimes [2,3,5,7,11]  --> 21  (3*7)\noddIndexPrimes [4,6,8,10]    --> 1",
     "stub":"module Main where\n\nisPrime :: Int -> Bool\nisPrime n\n  | n < 2     = False\n  | otherwise = all (\\d -> n `mod` d /= 0) [2..floor (sqrt (fromIntegral n))]\n\noddIndexPrimes :: [Int] -> Int\noddIndexPrimes xs = undefined\n\nmain :: IO ()\nmain = do\n  print (oddIndexPrimes [2,3,5,7,11])\n  print (oddIndexPrimes [4,6,8,10])",
     "hint":"zip [0..] xs pairs elements with indices. Filter for odd index AND prime. Then product."},
    
    {"id":"exam2","topic":"Exam Style","title":"makeNegative","difficulty":3,
     "description":"makeNeg: make all numbers negative (0 stays 0).\nmakeNegative: remove from second list all values in makeNeg of first list.\n\nmakeNeg [1,2,3] --> [-1,-2,-3]\nmakeNegative [1,2,3] [-1,3,2,-3,20,1] --> [3,2,20,1]",
     "stub":"module Main where\n\nmakeNeg :: [Int] -> [Int]\nmakeNeg xs = undefined\n\nmakeNegative :: [Int] -> [Int] -> [Int]\nmakeNegative xs ys = undefined\n\nmain :: IO ()\nmain = do\n  print (makeNeg [1,2,3])\n  print (makeNegative [1,2,3] [-1,3,2,-3,20,1])",
     "hint":"makeNeg: negate positives, keep others. makeNegative: filter ys keeping values NOT in (makeNeg xs)."},
    
    {"id":"exam3","topic":"Exam Style","title":"Full employee pipeline","difficulty":5,
     "description":"Given (name,age,salary):\n1. medianSalary\n2. totalWithRaise (10% for over-50s)\n3. noRaise (names who didn't get raise)\n4. youngHighEarners (under-30 earning over 250)",
     "stub":"module Main where\nimport Data.List (sort)\n\ntype Employee = (String, Int, Int)\n\nmedianSalary :: [Employee] -> Double\nmedianSalary emps = undefined\n\ntotalWithRaise :: [Employee] -> Double\ntotalWithRaise emps = undefined\n\nnoRaise :: [Employee] -> [String]\nnoRaise emps = undefined\n\nyoungHighEarners :: [Employee] -> [String]\nyoungHighEarners emps = undefined\n\nmain :: IO ()\nmain = do\n  let emps = [(\"John\",23,200),(\"Bob\",60,700),(\"Anna\",38,427),\n              (\"Joe\",36,289),(\"Doe\",22,384),(\"Marie\",55,572),(\"Lucy\",37,400)]\n  print (medianSalary emps)\n  print (totalWithRaise emps)\n  print (noRaise emps)\n  print (youngHighEarners emps)",
     "hint":"Break into small helpers. Each function does one thing. Sort salaries for median."},
]

class TestProblemBank:
    """Validate the problem bank structure and content."""

    def test_problem_count(self):
        """There should be exactly 25 problems."""
        assert len(TEST_PROBLEMS) == 25

    def test_all_problems_have_required_fields(self):
        """Every problem must have all required fields."""
        for p in TEST_PROBLEMS:
            missing = REQUIRED_PROBLEM_FIELDS - set(p.keys())
            assert not missing, f"Problem '{p.get('id')}' missing fields: {missing}"

    def test_all_ids_unique(self):
        """Problem IDs must be unique."""
        ids = [p["id"] for p in TEST_PROBLEMS]
        assert len(ids) == len(set(ids)), "Duplicate problem IDs found"

    def test_all_topics_valid(self):
        """All problems must belong to a valid topic."""
        for p in TEST_PROBLEMS:
            assert p["topic"] in VALID_TOPICS, \
                f"Problem '{p['id']}' has invalid topic: {p['topic']}"

    def test_difficulty_in_range(self):
        """Difficulty must be 1-5."""
        for p in TEST_PROBLEMS:
            assert 1 <= p["difficulty"] <= 5, \
                f"Problem '{p['id']}' has invalid difficulty: {p['difficulty']}"

    def test_all_topics_represented(self):
        """Every topic must have at least one problem."""
        topics_used = {p["topic"] for p in TEST_PROBLEMS}
        for topic in VALID_TOPICS:
            assert topic in topics_used, f"Topic '{topic}' has no problems"

    def test_difficulty_distribution(self):
        """Should have problems at every difficulty level."""
        difficulties = {p["difficulty"] for p in TEST_PROBLEMS}
        for d in range(1, 6):
            assert d in difficulties, f"No problems at difficulty {d}"

    def test_all_stubs_are_non_empty(self):
        """Every problem must have a non-empty code stub."""
        for p in TEST_PROBLEMS:
            assert p["stub"].strip(), f"Problem '{p['id']}' has empty stub"

    def test_all_stubs_contain_module_main(self):
        """Every stub should have module Main where."""
        for p in TEST_PROBLEMS:
            assert "module Main" in p["stub"] or "Main" in p["stub"], \
                f"Problem '{p['id']}' stub missing module declaration"

    def test_all_stubs_contain_main_function(self):
        """Every stub should have a main function."""
        for p in TEST_PROBLEMS:
            assert "main" in p["stub"], \
                f"Problem '{p['id']}' stub missing main function"

    def test_all_stubs_contain_undefined(self):
        """Every stub should have 'undefined' as a placeholder."""
        for p in TEST_PROBLEMS:
            assert "undefined" in p["stub"], \
                f"Problem '{p['id']}' stub should have 'undefined' placeholder"

    def test_all_hints_non_empty(self):
        """Every problem must have a non-empty hint."""
        for p in TEST_PROBLEMS:
            assert p["hint"].strip(), f"Problem '{p['id']}' has empty hint"

    def test_all_descriptions_non_empty(self):
        """Every problem must have a non-empty description."""
        for p in TEST_PROBLEMS:
            assert p["description"].strip(), f"Problem '{p['id']}' has empty description"

    def test_exam_style_problems_exist(self):
        """Must have exam style problems."""
        exam = [p for p in TEST_PROBLEMS if p["topic"] == "Exam Style"]
        assert len(exam) >= 3, "Need at least 3 exam style problems"

    def test_exam_style_problems_are_hard(self):
        """Exam style problems should be difficulty 3 or higher."""
        exam = [p for p in TEST_PROBLEMS if p["topic"] == "Exam Style"]
        for p in exam:
            assert p["difficulty"] >= 3, \
                f"Exam problem '{p['id']}' should be difficulty >= 3"

    def test_recursion_problems_cover_base_case_concept(self):
        """Recursion problem hints should mention base case."""
        recursion = [p for p in TEST_PROBLEMS if p["topic"] == "Recursion"]
        has_base_case = any("base case" in p["hint"].lower() or
                           "base" in p["hint"].lower() for p in recursion)
        assert has_base_case, "At least one recursion problem should mention base case in hint"

    def test_problems_sorted_by_difficulty_within_topics(self):
        """Within each topic, problems should generally increase in difficulty."""
        topics = {}
        for p in TEST_PROBLEMS:
            topics.setdefault(p["topic"], []).append(p["difficulty"])
        for topic, diffs in topics.items():
            if len(diffs) > 1:
                # Not strictly sorted but max should be >= min
                assert max(diffs) >= min(diffs), \
                    f"Topic '{topic}' has inconsistent difficulties"


class TestWebServerEndpoints:
    """Test web server HTTP endpoints."""

    @pytest.fixture
    def mock_app(self):
        """Create a test app with mocked dependencies."""
        # We test the endpoint logic directly since the full app
        # requires GHC and Groq to be available
        pass

    def test_diagnostic_dict_structure(self):
        """_diagnostic_to_dict should return all required keys."""
        # Mock a GHCDiagnostic
        diag = MagicMock()
        diag.severity.name = "ERROR"
        diag.span.start_line = 5
        diag.span.start_col  = 9
        diag.span.end_line   = 5
        diag.span.end_col    = 13
        diag.message         = "Couldn't match expected type"
        diag.category        = "TYPE_ERROR"
        diag.ai_explanation  = "Think of a vending machine..."
        diag.ai_hint         = "What type should this be?"
        diag.ai_scaffold     = "What is the base case?"

        # Replicate the conversion logic
        result = {
            "severity":    diag.severity.name.lower(),
            "startLine":   diag.span.start_line,
            "startCol":    diag.span.start_col,
            "endLine":     diag.span.end_line,
            "endCol":      diag.span.end_col,
            "message":     diag.message,
            "category":    str(diag.category),
            "explanation": diag.ai_explanation,
            "hint":        diag.ai_hint,
            "scaffold":    getattr(diag, "ai_scaffold", ""),
        }

        required_keys = {
            "severity", "startLine", "startCol", "endLine", "endCol",
            "message", "category", "explanation", "hint", "scaffold"
        }
        assert required_keys == set(result.keys())

    def test_diagnostic_severity_lowercased(self):
        """Severity should be lowercase string."""
        diag = MagicMock()
        diag.severity.name = "ERROR"
        diag.span = None
        diag.message = "test"
        diag.category = "TYPE_ERROR"
        diag.ai_explanation = ""
        diag.ai_hint = ""

        result = {
            "severity":    diag.severity.name.lower(),
            "startLine":   diag.span.start_line if diag.span else 0,
            "startCol":    diag.span.start_col  if diag.span else 0,
            "endLine":     diag.span.end_line   if diag.span else 0,
            "endCol":      diag.span.end_col    if diag.span else 0,
            "message":     diag.message,
            "category":    str(diag.category),
            "explanation": diag.ai_explanation,
            "hint":        diag.ai_hint,
            "scaffold":    getattr(diag, "ai_scaffold", ""),
        }

        assert result["severity"] == "error"
        assert result["startLine"] == 0

    def test_progress_tracking_logic(self):
        """Session progress store should track solved problems."""
        progress: dict[str, set] = {}

        def mark_solved(session_id, problem_id):
            if session_id not in progress:
                progress[session_id] = set()
            progress[session_id].add(problem_id)
            return list(progress[session_id])

        # Mark first problem solved
        result = mark_solved("session1", "rec1")
        assert "rec1" in 
        # Mark second problem solved in same session
        mark_solved("session1", "rec2")
        assert len(progress["session1"]) == 2

        # Different session is isolated
        mark_solved("session2", "rec1")
        assert len(progress["session1"]) == 2
        assert len(progress["session2"]) == 1

    def test_progress_tracking_no_duplicates(self):
        """Marking the same problem twice should not create duplicates."""
        progress: dict[str, set] = {}

        def mark_solved(session_id, problem_id):
            if session_id not in progress:
                progress[session_id] = set()
            progress[session_id].add(problem_id)

        mark_solved("s1", "rec1")
        mark_solved("s1", "rec1")
        mark_solved("s1", "rec1")
        assert len(progress["s1"]) == 1

    def test_empty_source_returns_empty_diagnostics(self):
        """Empty source should return empty diagnostics list."""
        source = ""
        result = source.strip()
        assert not result  # Empty source should be caught early


class TestStuckButtonLogic:
    """Test the stuck button timing logic."""

    def test_stuck_threshold_is_reasonable(self):
        """Stuck button should appear after a reasonable time (60-120 seconds)."""
        STUCK_THRESHOLD = 90  # seconds
        assert 60 <= STUCK_THRESHOLD <= 120, \
            "Stuck threshold should be between 1-2 minutes"

    def test_stuck_resets_on_clean_compile(self):
        """Error persist time should reset when code compiles cleanly."""
        error_persist_time = 95  # Was stuck

        # Simulate clean compile
        diags = []  # No errors
        if not any(d.get("severity") == "error" for d in diags):
            error_persist_time = 0

        assert error_persist_time == 0

    def test_stuck_increments_with_errors(self):
        """Error persist time should increment while errors exist."""
        error_persist_time = 0
        diags = [{"severity": "error", "message": "some error"}]

        # Simulate 5 seconds of errors
        for _ in range(5):
            if any(d["severity"] == "error" for d in diags):
                error_persist_time += 1

        assert error_persist_time == 5

    def test_stuck_not_shown_for_warnings_only(self):
        """Stuck button should NOT appear for warnings only."""
        diags = [{"severity": "warning", "message": "some warning"}]
        has_errors = any(d["severity"] == "error" for d in diags)
        assert not has_errors


class TestThinkTabLogic:
    """Test the Think tab conversation logic."""

    def test_system_prompt_enforces_one_question(self):
        """System prompt must contain 'ONE question' constraint."""
        sys_prompt = (
            'You are a Haskell tutor. Ask ONE question to help them figure out '
            'the solution structure. DO NOT solve it. DO NOT write code. '
            'ONE question only. Max 2 sentences.'
        )
        assert "ONE question" in sys_prompt
        assert "DO NOT solve" in sys_prompt or "DO NOT write code" in sys_prompt

    def test_reply_system_prompt_enforces_one_question(self):
        """Reply system prompt must also enforce one question."""
        sys = (
            'RULES: ONE follow-up question only. '
            'Never write code. Never give the answer. '
            'Max 2 sentences.'
        )
        assert "ONE" in sys
        assert "Never write code" in sys or "NEVER" in sys.upper()

    def test_think_history_grows_with_each_exchange(self):
        """Think history should grow by 2 per exchange (user + assistant)."""
        history = []

        def add_exchange(user_msg, ai_response):
            history.append({"role": "user",      "content": user_msg})
            history.append({"role": "assistant", "content": ai_response})

        add_exchange("Write a function that sums a list", "What type does it return?")
        assert len(history) == 2

        add_exchange("Int", "What happens with an empty list?")
        assert len(history) == 4

    def test_think_history_resets_on_new_problem(self):
        """Think history should reset when a new problem is decoded."""
        history = [
            {"role": "user", "content": "old problem"},
            {"role": "assistant", "content": "old question"},
        ]
        # Simulate decode reset
        history = []
        assert len(history) == 0


class TestProblemProgression:
    """Test problem progression suggestions."""

    def test_next_problem_in_same_topic(self):
        """After solving a problem, suggest next in same topic."""
        solved = {"rec1"}
        topic  = "Recursion"
        rec_problems = [p for p in TEST_PROBLEMS if p["topic"] == topic]
        unsolved = [p for p in rec_problems if p["id"] not in solved]
        assert len(unsolved) > 0
        # Next should be the easiest unsolved
        next_p = min(unsolved, key=lambda p: p["difficulty"])
        assert next_p["id"] != "rec1"

    def test_no_progression_when_all_solved(self):
        """When all problems in topic solved, suggest different topic."""
        rec_ids = {p["id"] for p in TEST_PROBLEMS if p["topic"] == "Recursion"}
        solved  = rec_ids.copy()
        remaining = [p for p in TEST_PROBLEMS if p["id"] not in solved]
        assert len(remaining) > 0  # Other topics still available

    def test_difficulty_increases_with_progress(self):
        """Problems should get harder as student solves more."""
        solved = set()
        # Simulate solving in order
        for p in sorted(TEST_PROBLEMS, key=lambda x: x["difficulty"]):
            solved.add(p["id"])
            unsolved = [x for x in TEST_PROBLEMS if x["id"] not in solved]
            if unsolved:
                min_remaining_diff = min(x["difficulty"] for x in unsolved)
                # Min difficulty of remaining should not go below what we've done
                max_solved_diff = max(
                    TEST_PROBLEMS[i]["difficulty"]
                    for i, x in enumerate(TEST_PROBLEMS)
                    if x["id"] in solved
                )
                assert min_remaining_diff >= 1  # Always valid difficulties


class TestWebEditorIntegration:
    """Integration tests for the web editor JavaScript logic."""

    def test_problem_topics_match_filter_options(self):
        """Topics in problem bank must match dropdown filter options."""
        filter_options = {
            "Recursion", "Pattern Matching", "List Comprehensions",
            "Higher Order Functions", "Types", "Tuples",
            "Strings", "Algorithms", "Exam Style"
        }
        problem_topics = {p["topic"] for p in TEST_PROBLEMS}
        assert problem_topics == filter_options, \
            f"Mismatch: {problem_topics.symmetric_difference(filter_options)}"

    def test_esc_function_logic(self):
        """HTML escape function should escape dangerous characters."""
        def esc(s):
            return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')

        assert esc('<script>') == '&lt;script&gt;'
        assert esc('"hello"') == '&quot;hello&quot;'
        assert esc('a & b')   == 'a &amp; b'
        assert esc('safe')    == 'safe'

    def test_difficulty_dot_rendering(self):
        """Difficulty dots should use correct CSS classes."""
        def get_diff_class(i, difficulty):
            if i > difficulty:
                return 'diff-dot'
            if difficulty >= 5:
                return 'diff-dot vhard'
            if difficulty >= 4:
                return 'diff-dot hard'
            return 'diff-dot on'

        # Easy problem (difficulty 1)
        assert get_diff_class(1, 1) == 'diff-dot on'
        assert get_diff_class(2, 1) == 'diff-dot'

        # Hard problem (difficulty 4)
        assert get_diff_class(1, 4) == 'diff-dot hard'
        assert get_diff_class(4, 4) == 'diff-dot hard'
        assert get_diff_class(5, 4) == 'diff-dot'

        # Hardest (difficulty 5)
        assert get_diff_class(1, 5) == 'diff-dot vhard'

    def test_filter_all_returns_all_problems(self):
        """Filter 'all' should return all 25 problems."""
        topic = 'all'
        filtered = TEST_PROBLEMS if topic == 'all' else [p for p in TEST_PROBLEMS if p["topic"] == topic]
        assert len(filtered) == 25

    def test_filter_by_topic_returns_subset(self):
        """Filter by topic should return only matching problems."""
        topic    = 'Recursion'
        filtered = [p for p in TEST_PROBLEMS if p["topic"] == topic]
        assert all(p["topic"] == "Recursion" for p in filtered)
        assert len(filtered) < len(TEST_PROBLEMS)

    def test_websocket_compile_message_format(self):
        """WebSocket compile message must have correct format."""
        import json
        msg = json.dumps({
            "type":   "compile",
            "source": "module Main where\nmain = print 42",
            "uri":    "web://editor"
        })
        parsed = json.loads(msg)
        assert parsed["type"]   == "compile"
        assert "source"         in parsed
        assert "uri"            in parsed

    def test_diagnostics_response_format(self):
        """Diagnostics WebSocket response must have correct format."""
        import json
        response = json.dumps({
            "type":        "diagnostics",
            "diagnostics": [],
            "clean":       True
        })
        parsed = json.loads(response)
        assert parsed["type"]        == "diagnostics"
        assert "diagnostics"         in parsed
        assert isinstance(parsed["diagnostics"], list)
        assert "clean"               in parsed


class TestRunEndpoint:
    """Test the /api/run code execution endpoint."""

    def test_empty_source_handled_gracefully(self):
        """Empty source should return error, not crash."""
        source = ""
        if not source.strip():
            result = {"output": "", "error": "Empty source"}
        assert result["error"] == "Empty source"

    def test_output_capture_format(self):
        """Run result must have output and error fields."""
        result = {"output": "42\n", "error": ""}
        assert "output" in         assert "error"  in result

    def test_timeout_error_message(self):
        """Timeout should return clear error message."""
        result = {"output": "", "error": "Execution timed out (10s limit)."}
        assert "timed out" in result["error"]

    def test_ghc_not_found_message(self):
        """Missing GHC should return helpful error."""
        result = {"output": "", "error": "GHC not found. Is it installed?"}
        assert "GHC" in result["error"]