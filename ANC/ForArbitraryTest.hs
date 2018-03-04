module ForArbitraryTest where 

import ForParser
import Test.QuickCheck

arbitraryIdentifier :: Gen String
arbitraryIdentifier = do firstChar <- elements $ ['a' .. 'z'] ++ ['A' .. 'Z']
                         return [firstChar]

arbitraryConstant :: Gen Int
arbitraryConstant = do firstValue <- elements $ [1 .. 20]
					   return [firstValue]

arbitrarySizedExprFor :: Int -> Gen ExprFor
arbitrarySizedExprFor n | n <= 0 = frequency [(1, Var <$> arbitraryIdentifier)
											 ,(1, Const <$> arbitraryConstant)]
                                 | otherwise = frequency [(1, Plus <$> smallerArbitrary <*> smallerArbitrary) 
                                                         ,(1, Minus <$> smallerArbitrary <*> smallerArbitrary)]
                                        where smallerArbitrary = arbitrarySizedExprFor (n `div` 2)