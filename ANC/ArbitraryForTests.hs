module ArbitraryForTests where 

import ForParser
import ForLambdaCommon

import Test.QuickCheck

-- 
-- ArbitraryForTests.hs
-- A set of functions for generating arbitrary For programs.
-- 

data Difficulty = Debug | Easy | Medium | Hard | VeryHard deriving (Read, Show)

-- 
-- Generators
-- 

combineStringWithNumber :: String -> Int -> String
combineStringWithNumber s i = s ++ show i

arbitraryIdentifier :: Int -> Gen String
arbitraryIdentifier upperBound = do identifier <- elements $ map (combineStringWithNumber "a") [0 .. upperBound]
                                    return identifier
                                            
arbitraryConstant :: Gen Int
arbitraryConstant = do firstInt <- elements $ [0 .. 10]
                       return firstInt 

arbitraryConstantExpression :: Gen Expr
arbitraryConstantExpression = Const <$> arbitraryConstant

arbitrarySizedExpr :: Int -> Int -> Gen Expr
arbitrarySizedExpr upperBound n | n <= 0 = if upperBound == -1 
                                           then Const <$> arbitraryConstant
                                           else frequency [(1, Var <$> arbitraryIdentifier upperBound), (1, Const <$> arbitraryConstant)]
                                | otherwise = if upperBound == -1 
                                              then frequency [(1, Const <$> arbitraryConstant)
                                                             ,(1, Plus <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2)) 
                                                             ,(1, Minus <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2))] 
                                              else frequency [(1, Var <$> arbitraryIdentifier upperBound)
                                                             ,(1, Const <$> arbitraryConstant)
                                                             ,(1, Plus <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2)) 
                                                             ,(1, Minus <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2))] 

instance Arbitrary Expr where
    arbitrary = sized (arbitrarySizedExpr 10)

    shrink n@(Plus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n@(Minus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n = [n]

arbitrarySizedCmp :: Int -> Int -> Gen Cmp
arbitrarySizedCmp upperBound n = frequency [(1, Equal <$> smallerArbitrary <*> smallerArbitrary)
                                           ,(1, Le <$> smallerArbitrary <*> smallerArbitrary)
                                           ,(1, Ge <$> smallerArbitrary <*> smallerArbitrary)]
                                 where smallerArbitrary = arbitrarySizedExpr upperBound ((n `div` 3) - 1)

instance Arbitrary Cmp where
  arbitrary = sized (arbitrarySizedCmp 10)

  shrink n = [n]

-- 
-- Language Specific: For
-- 

-- If statements may be able to declare new variables for use outside blocks
-- Here, we do not allow variables defined within the If to be used outside the block
arbitrarySizedProgForIf :: [Int] -> Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgForIf frequencies upperBound n = do comp <- arbitrarySizedCmp upperBound ((n `div` 3) - 1)
                                                      (if_body, _) <- arbitrarySizedProgFor frequencies upperBound ((n `div` 3) - 2)
                                                      (else_body, _) <- arbitrarySizedProgFor frequencies upperBound ((n `div` 3) - 2)
                                                      return (If comp if_body else_body, upperBound)

-- Seq statements may be able to declare new variables for use in next block
arbitrarySizedProgForSeq :: [Int] -> Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgForSeq frequencies upperBound n = do (oneHalfArbitrarySizedProgFor1, firstBlockUpperBound) <- arbitrarySizedProgFor frequencies upperBound (n `div` 2)
                                                       (oneHalfArbitrarySizedProgFor2, secondBlockUpperBound) <- arbitrarySizedProgFor frequencies firstBlockUpperBound ((n `div` 2) - 1)
                                                       let arbitrarySeq = Seq oneHalfArbitrarySizedProgFor1 oneHalfArbitrarySizedProgFor2
                                                       return (arbitrarySeq, secondBlockUpperBound)

-- For statments may be able to declare new variables for use in outside blocks
-- Here, we do not allow variables defined within the For to be used outside the block
arbitrarySizedProgForFor :: [Int] -> Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgForFor frequencies upperBound n = do (variableName, upperBoundLoop) <- (if upperBound == -1 then return ("a0", 0) else if upperBound == 10 then flip (,) 10 <$> arbitraryIdentifier upperBound else frequency [(4,  return ("a" ++ show (upperBound + 1), upperBound + 1)), (1, (,) <$> arbitraryIdentifier upperBound <*> return upperBound)])
                                                       initialize <- arbitrarySizedExpr upperBound ((n `div` 4) - 1)
                                                       comp <- arbitrarySizedCmp upperBoundLoop (n `div` 4)
                                                       expr <- arbitrarySizedExpr upperBoundLoop (n `div` 4)
                                                       (body, _) <- arbitrarySizedProgFor frequencies upperBoundLoop ((n `div` 4) - 1)
                                                       return (For variableName initialize comp expr body, upperBound)

arbitrarySizedProgAssign :: Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgAssign upperBound n = if upperBound == -1 then do expr <- arbitrarySizedExpr upperBound 3
                                                                    return (Assign "a0" expr, 0)
                                     else if upperBound == 10 then do variableName <- arbitraryIdentifier upperBound
                                                                      expr <- arbitrarySizedExpr upperBound 3
                                                                      return (Assign variableName expr, 10)
                                     else do reassignedVariableName <- arbitraryIdentifier upperBound
                                             expr <- arbitrarySizedExpr upperBound 3
                                             frequency [(5, return (Assign reassignedVariableName expr, upperBound) )                   {- Assign to previoiusly declared -}
                                                       ,(1, return (Assign ("a" ++ show (upperBound + 1)) expr, upperBound + 1) )]      {- Possibly declare new variable  -}

-- Generator for general For programs
-- Parameters:
-- frequencies, the frequency of an Assign, If, For, and Seq statement
-- upperBound, the maximum int in the declared variables so far
-- n, the number of terms (size) of the desired ProgFor program
arbitrarySizedProgFor :: [Int] -> Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgFor frequencies@[freqAssign, freqIf, freqFor, freqSeq] upperBound n | n <= 0 = arbitrarySizedProgAssign upperBound n
                                                                                      | otherwise = frequency [(freqAssign, arbitrarySizedProgAssign upperBound n)
                                                                                                              ,(freqIf, arbitrarySizedProgForIf frequencies upperBound n)
                                                                                                              ,(freqFor, arbitrarySizedProgForFor frequencies upperBound n)
                                                                                                              ,(freqSeq, arbitrarySizedProgForSeq frequencies upperBound n)] 
arbitrarySizedProgFor _ upperBound n = error "Only four frequencies should be provided."

arbitrarySizedProgForWithDifficulty :: Difficulty -> Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgForWithDifficulty Debug _ _ = do constant <- arbitraryConstantExpression
                                                   frequency [(1, return (Assign "a1" constant, 1))]
arbitrarySizedProgForWithDifficulty Easy upperBound n = arbitrarySizedProgFor [1, 0, 0, 0] upperBound n
arbitrarySizedProgForWithDifficulty Medium upperBound n = arbitrarySizedProgFor [1, 1, 0, 0] upperBound n
arbitrarySizedProgForWithDifficulty Hard upperBound n = arbitrarySizedProgFor [2, 1, 1, 0] upperBound n
arbitrarySizedProgForWithDifficulty VeryHard upperBound n = arbitrarySizedProgFor [2, 1, 1, 1] upperBound n

instance Arbitrary ProgFor where
  arbitrary = fst <$> sized (arbitrarySizedProgFor [1, 1, 1, 1] (-1))

  shrink n@(If c1 p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n@(For _ e1 c1 e2 p1) = shrink p1 ++ [n]
  shrink n@(Seq p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n = [n]



