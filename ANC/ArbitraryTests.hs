module ArbitraryTests where 

import ForParser
import LambdaParser
import ForLambdaCommon
import Test.QuickCheck

-- 
-- ArbitraryTests.hs
-- A set of functions for generating arbitrary For and ProgLambda programs.
-- 

arbitraryIdentifier :: Gen String
arbitraryIdentifier = do firstChar <- elements $ ['a' .. 'z']
                         return [firstChar]

arbitraryConstant :: Gen Int
arbitraryConstant = do firstInt <- elements $ [1 .. 10]
                       return firstInt

arbitrarySizedExpr :: Int -> Gen Expr
arbitrarySizedExpr n | n <= 0 = frequency [(1, Var <$> arbitraryIdentifier)
                                          ,(1, Const <$> arbitraryConstant)]
                                 | otherwise = frequency [(1, Var <$> arbitraryIdentifier)
                                                         ,(1, Const <$> arbitraryConstant)
                                                         ,(1, Plus <$> smallerArbitrary <*> smallerArbitrary) 
                                                         ,(1, Minus <$> smallerArbitrary <*> smallerArbitrary)]
                                               where smallerArbitrary = arbitrarySizedExpr (n `div` 2)

instance Arbitrary Expr where
    arbitrary = sized arbitrarySizedExpr

    shrink n@(Plus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n@(Minus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n = [n]

arbitrarySizedCmp :: Int -> Gen Cmp
arbitrarySizedCmp n = frequency [(1, Equal <$> smallerArbitrary <*> smallerArbitrary)
                                ,(1, Le <$> smallerArbitrary <*> smallerArbitrary)
                                ,(1, Ge <$> smallerArbitrary <*> smallerArbitrary)]
                      where smallerArbitrary = arbitrarySizedExpr (n `div` 2)

instance Arbitrary Cmp where
  arbitrary = sized arbitrarySizedCmp

  shrink n = [n]

arbitrarySizedProgFor :: Int -> Gen ProgFor
arbitrarySizedProgFor n | n <= 0 = Assign <$> arbitraryIdentifier <*> arbitrarySizedExpr (n - 1)
                                 | otherwise = frequency [(1, Assign <$> arbitraryIdentifier <*> arbitrarySizedExpr (n - 1))
                                                         ,(1, If <$> arbitrarySizedCmp (n `div` 3) <*> arbitrarySizedProgFor (n `div` 3) <*> arbitrarySizedProgFor (n `div` 3))
                                                         ,(1, For <$> arbitraryIdentifier <*> arbitrarySizedExpr (n `div` 4) <*> arbitrarySizedCmp (n `div` 4) <*> arbitrarySizedExpr (n `div` 4) <*> arbitrarySizedProgFor (n `div` 4))
                                                         ,(1, Seq <$> arbitrarySizedProgFor (n `div` 2) <*> arbitrarySizedProgFor (n `div` 2))] 

instance Arbitrary ProgFor where
  arbitrary = sized arbitrarySizedProgFor

  shrink n@(If c1 p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n@(For _ e1 c1 e2 p1) = shrink p1 ++ [n]
  shrink n@(Seq p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n = [n]

arbitrarySizedProgLambda :: Int -> Gen ProgLambda
arbitrarySizedProgLambda n | n <= 0 = return UnitLambda 
                                    | otherwise = frequency [(1, IfL <$> arbitrarySizedCmp (n `div` 3) <*> arbitrarySizedProgLambda (n `div` 3) <*> arbitrarySizedProgLambda (n `div` 3))
                                                            ,(1, ExprL <$> arbitrarySizedExpr (n - 1))
                                                            ,(1, LetLambda <$> arbitraryIdentifier <*> arbitrarySizedProgLambda (n `div` 2) <*> arbitrarySizedProgLambda (n `div` 2))
                                                            ,(1, LetRecLambda <$> arbitraryIdentifier <*> arbitraryIdentifier <*> arbitrarySizedProgLambda (n `div` 2) <*> arbitrarySizedProgLambda (n `div` 2))
                                                            ,(1, App <$> arbitrarySizedApp (n - 1))]

instance Arbitrary ProgLambda where
  arbitrary = sized arbitrarySizedProgLambda

  shrink n@(IfL _ p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n@(LetLambda _ p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n@(LetRecLambda _ _ p1 p2) = shrink p1 ++ shrink p2 ++ [n]
  shrink n = [n]

arbitrarySizedApp :: Int -> Gen App
arbitrarySizedApp n | n <= 0 = SimpleApp <$> arbitraryIdentifier <*> arbitrarySizedExpr (n - 1)
                             | otherwise = frequency [(1, SimpleApp <$> arbitraryIdentifier <*> arbitrarySizedExpr (n - 1))
                                                     ,(1, ComplexApp <$> arbitrarySizedApp (n `div` 2) <*> arbitrarySizedExpr (n `div` 2))]

instance Arbitrary App where
  arbitrary = sized arbitrarySizedApp

  shrink n@(ComplexApp a _) = shrink a ++ [n]
  shrink n = [n]



