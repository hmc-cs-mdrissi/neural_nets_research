module ArbitraryTests where 

import ForParser
import Test.QuickCheck

arbitraryIdentifier :: Gen String
arbitraryIdentifier = do firstChar <- elements $ ['a' .. 'z'] ++ ['A' .. 'Z']
                         return [firstChar]

arbitraryConstant :: Gen Integer
arbitraryConstant = do firstInt <- elements $ [1 .. 10]
                       return firstInt

arbitrarySizedExpr :: Int -> Gen Expr
arbitrarySizedExpr n | n <= 0 = frequency [(1, Var <$> arbitraryIdentifier)
                                          ,(1, Const <$> arbitraryConstant)]
                                 | otherwise = frequency [(1, Var <$> arbitraryIdentifier)
                                                         ,(1, Const <$> arbitraryConstant)
                                                         ,(1, Plus <$> smallerArbitrary <*> smallerArbitrary) 
                                                         ,(1, Minus <$> smallerArbitrary <*> smallerArbitrary)]
                                               where smallerArbitrary = arbitrarySizedExprFor (n `div` 2)

instance Arbitrary Expr where
    arbitrary = sized arbitrarySizedExpr

    shrink n@(Plus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n@(Minus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n = [n]

--data Expr = Var String | Const Integer | Plus Expr Expr | Minus Expr Expr deriving (Show, Generic)
--data Cmp = Equal Expr Expr | Le Expr Expr | Ge Expr Expr deriving (Show, Generic)

--arbitrarySizedCmp :: Int -> Gen Cmp
--arbitrarySizedCmp n = frequency [(1, EqualFor <$> smallerArbitrary <*> smallerArbitrary)
--                                ,(1, LeFor <$> smallerArbitrary <*> smallerArbitrary)
--                                ,(1, GeFor <$> smallerArbitrary <*> smallerArbitrary)]
--                      where smallerArbitrary = arbitrarySizedExprFor (n `div` 2)

--instance Arbitrary Cmp where
--  arbitrary = sized arbitrarySizedCmp

--  shrink n = [n]

--arbitrarySizedProgFor :: Int -> Gen ProgFor
--arbitrarySizedProgFor n | n <= 0 = Assign <$> arbitraryIdentifier <*> arbitrarySizedExprFor (n - 1)
--                                 | otherwise = frequency[(1, Assign <$> arbitraryIdentifier <*> arbitrarySizedExprFor (n - 1))
--                                                        ,(1, If <$> arbitrarySizedCmp (n `div` 3) <*> arbitrarySizedProgFor (n `div` 3) <*> arbitrarySizedProgFor (n `div` 3))
--                                                        ,(1, For <$> arbitraryIdentifier <*> arbitrarySizedExprFor (n `div` 4) <*> arbitrarySizedCmp (n `div` 4) <*> arbitrarySizedExprFor (n `div` 4) <*> arbitrarySizedProgFor (n `div` 4))
--                                                        ,(1, Seq <$> arbitrarySizedProgFor (n `div` 2) <*> arbitrarySizedProgFor (n `div` 2))] 

--instance Arbitrary ProgFor where
--  arbitrary = sized arbitrarySizedProgFor

--  shrink n@(If c1 p1 p2) = shrink p1 ++ shrink p2 ++ [n]
--  shrink n@(For _ e1 c1 e2 p1) = shrink p1 ++ [n]
--  shrink n@(Seq p1 p2) = shrink p1 ++ shrink p2 ++ [n]
--  shrink n = [n]




