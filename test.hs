-- Write a function that takes a non-negative integer and returns
-- the number of odd digits in that integer.
-- If the input is negative, return an error message
-- as a string "Negative number error".
-- Example:
-- countOdds 1234 -- 2
-- countOdds 987654321 -- 5


countOdds :: Int -> Int

countOdds x
  | x < 0 = error "negative number error"
  | (x `mod` 10) `mod` 2 == 1 = 1 + countOdds (x `div` 10)
  | otherwise = countOdds (x `div` 10)

-- main = print(countOdds 1234) -- 2
-- main = print(countOdds 0) -- 0
main = print(countOdds 987654321) -- 5
-- main = print(countOdds (-5678)) -- "Negative number error"