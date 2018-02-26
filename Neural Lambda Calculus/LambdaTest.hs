module LambdaTest where

import LambdaParser

import Test.QuickCheck

arbitraryIdentifier :: Gen String
arbitraryIdentifier = do firstChar <- elements $ ['a' .. 'z'] ++ ['A' .. 'Z']
                         return [firstChar]


arbitrarySizedLambaExpression :: Int -> Gen LambdaExpression
arbitrarySizedLambaExpression n | n <= 0 = Variable <$> arbitraryIdentifier
                                 | otherwise = frequency [(2, Variable <$> arbitraryIdentifier)
                                                         ,(1, Application <$> smallerArbitrary <*> smallerArbitrary) 
                                                         ,(1, Abstraction <$> arbitraryIdentifier <*> smallerArbitrary)]
                                        where smallerArbitrary = arbitrarySizedLambaExpression (n `div` 2)

instance Arbitrary LambdaExpression where
    arbitrary = sized arbitrarySizedLambaExpression

    shrink (Application x y) = shrink x ++ shrink y
    shrink (Abstraction x expr) = shrink expr ++ (Abstraction x <$> shrink expr)
    shrink var@(Variable _) = [var]

prop_parse_show_weak_inverse :: LambdaExpression -> Bool
prop_parse_show_weak_inverse x = (parseLambda . show $ x) == Right x
