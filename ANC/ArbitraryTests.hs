module ArbitraryTests where 

import ForParser
import LambdaParser
import ForLambdaCommon

import System.Random
import Test.QuickCheck

-- 
-- ArbitraryTests.hs
-- A set of functions for generating arbitrary For and ProgLambda programs.
-- 

-- 
-- Generators
-- 

combineStringWithNumber :: String -> Int -> String
combineStringWithNumber s i = s ++ show i

arbitraryIdentifier :: Int -> Gen String
arbitraryIdentifier upperBound = do identifier <- elements $ map (combineStringWithNumber "a") [1 .. upperBound]
                                    return identifier
                                            
arbitraryConstant :: Gen Int
arbitraryConstant = do firstInt <- elements $ [0 .. 10]
                       return firstInt 

arbitrarySizedExpr :: Int -> Int -> Gen Expr
arbitrarySizedExpr upperBound n | n <= 0 = if upperBound == 0 
                                           then Const <$> arbitraryConstant
                                           else frequency [(1, Var <$> arbitraryIdentifier upperBound), (1, Const <$> arbitraryConstant)]
                                | otherwise = if upperBound == 0 
                                              then frequency [(1, Const <$> arbitraryConstant)
                                                             ,(1, Plus <$> arbitrarySizedExpr upperBound (n `div` 2) <*> arbitrarySizedExpr upperBound (n `div` 2)) 
                                                             ,(1, Minus <$> arbitrarySizedExpr upperBound (n `div` 2) <*> arbitrarySizedExpr upperBound (n `div` 2))] 
                                              else frequency [(1, Var <$> arbitraryIdentifier upperBound)
                                                             ,(1, Const <$> arbitraryConstant)
                                                             ,(1, Plus <$> arbitrarySizedExpr upperBound (n `div` 2) <*> arbitrarySizedExpr upperBound (n `div` 2)) 
                                                             ,(1, Minus <$> arbitrarySizedExpr upperBound (n `div` 2) <*> arbitrarySizedExpr upperBound (n `div` 2))] 

instance Arbitrary Expr where
    arbitrary = sized (arbitrarySizedExpr 10)

    shrink n@(Plus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n@(Minus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n = [n]

arbitrarySizedCmp :: Int -> Int -> Gen Cmp
arbitrarySizedCmp upperBound n = frequency [(1, Equal <$> smallerArbitrary <*> smallerArbitrary)
                                           ,(1, Le <$> smallerArbitrary <*> smallerArbitrary)
                                           ,(1, Ge <$> smallerArbitrary <*> smallerArbitrary)]
                                 where smallerArbitrary = arbitrarySizedExpr upperBound (n `div` 2) 

instance Arbitrary Cmp where
  arbitrary = sized (arbitrarySizedCmp 10)

  shrink n = [n]

-- 
-- Language Specific: For
-- 

-- If statements may be able to declare new variables for use outside blocks
-- Here, we do not allow variables defined within the If to be used outside the block
arbitrarySizedProgForIf :: Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgForIf upperBound n = do comp <- arbitrarySizedCmp upperBound (n `div` 3)
                                          (if_body, _) <- arbitrarySizedProgFor upperBound (n `div` 3)
                                          (else_body, _) <- arbitrarySizedProgFor upperBound (n `div` 3)
                                          return (If comp if_body else_body, upperBound)

-- Seq statements may be able to declare new variables for use in next block
arbitrarySizedProgForSeq :: Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgForSeq upperBound n = do (oneHalfArbitrarySizedProgFor1, firstBlockUpperBound) <- arbitrarySizedProgFor upperBound (n `div` 2)
                                           (oneHalfArbitrarySizedProgFor2, secondBlockUpperBound) <- arbitrarySizedProgFor firstBlockUpperBound (n `div` 2) 
                                           let arbitrarySeq = Seq oneHalfArbitrarySizedProgFor1 oneHalfArbitrarySizedProgFor2
                                           return (arbitrarySeq, secondBlockUpperBound)

-- For statments may be able to declare new variables for use in outside blocks
-- Here, we do not allow variables defined within the For to be used outside the block
arbitrarySizedProgForFor :: Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgForFor upperBound n = do (variableName, upperBoundLoop) <- (if upperBound == 0 then return ("a1", 1) else if upperBound == 10 then flip (,) 10 <$> arbitraryIdentifier upperBound else frequency [(4,  return ("a" ++ show (upperBound + 1), upperBound + 1)), (1, (,) <$> arbitraryIdentifier upperBound <*> return upperBound)])
                                           initialize <- arbitrarySizedExpr upperBound (n `div` 4)
                                           comp <- arbitrarySizedCmp upperBoundLoop (n `div` 4)
                                           expr <- arbitrarySizedExpr upperBoundLoop (n `div` 4)
                                           (body, _) <- arbitrarySizedProgFor upperBoundLoop (n `div` 4)
                                           return (For variableName initialize comp expr body, upperBound)

arbitrarySizedProgAssign :: Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgAssign upperBound n = if upperBound == 0 then do expr <- arbitrarySizedExpr upperBound (n - 1)
                                                                   return (Assign "a1" expr, 1)
                                     else if upperBound == 10 then do variableName <- arbitraryIdentifier upperBound
                                                                      expr <- arbitrarySizedExpr upperBound (n - 1)
                                                                      return (Assign variableName expr, 10)
                                     else do reassignedVariableName <- arbitraryIdentifier upperBound
                                             expr <- arbitrarySizedExpr upperBound (n - 1)
                                             frequency [(5, return (Assign reassignedVariableName expr, upperBound) )                   {- Assign to previoiusly declared -}
                                                       ,(1, return (Assign ("a" ++ show (upperBound + 1)) expr, upperBound + 1) )]      {- Possibly declare new variable  -}

-- Generator for general For programs
arbitrarySizedProgFor :: Int -> Int -> Gen (ProgFor, Int)
arbitrarySizedProgFor upperBound n | n <= 0 = arbitrarySizedProgAssign upperBound n
                                   | otherwise = frequency [(2, arbitrarySizedProgAssign upperBound n)
                                                           ,(1, arbitrarySizedProgForIf upperBound n)
                                                           ,(1, arbitrarySizedProgForFor upperBound n)
                                                           ,(1, arbitrarySizedProgForSeq upperBound n)] 
                                                      
instance Arbitrary ProgFor where
  arbitrary = fst <$> sized (arbitrarySizedProgFor 0)

  shrink n@(If c1 p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n@(For _ e1 c1 e2 p1) = shrink p1 ++ [n]
  shrink n@(Seq p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n = [n]



