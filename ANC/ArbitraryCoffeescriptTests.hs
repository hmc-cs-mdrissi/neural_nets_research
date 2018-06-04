{-# LANGUAGE TupleSections, DeriveGeneric #-}

module ArbitraryCoffeescriptTests where 

import Test.QuickCheck
import GHC.Generics
import Control.Arrow (first)
import Data.Aeson

-- 
-- ArbitraryCoffeescriptTests.hs
-- A set of functions for generating arbitrary coffee script programs.
-- 

data ProgramLength = Short | Long deriving (Read, Show)
data VariableVariety = Easy | Hard deriving (Read, Show)

data ExprCS = Var String | Const Int | Plus ExprCS ExprCS | Times ExprCS ExprCS | Equal ExprCS ExprCS deriving (Read, Show, Generic)
data SimpleCS = Assign String ExprCS | Expr ExprCS deriving (Read, Show, Generic)
data IfShortCS = IfSimple SimpleCS ExprCS | IfComplex IfShortCS ExprCS deriving (Read, Show, Generic)
data WhileShortCS = WhileSimple SimpleCS ExprCS | WhileComplex WhileShortCS ExprCS deriving (Read, Show, Generic)
data StatementShortCS = SimpleStatement SimpleCS | SimpleIf IfShortCS | SimpleWhile WhileShortCS deriving (Read, Show, Generic)
data StatementCS = ShortStatementCS StatementShortCS | If ExprCS CoffeeScript | While ExprCS CoffeeScript | IfElse ExprCS CoffeeScript CoffeeScript 
                    | IfThenElse ExprCS StatementShortCS StatementShortCS deriving (Read, Show, Generic)
data CoffeeScript = SimpleCS StatementCS | ComplexCS CoffeeScript StatementCS deriving (Read, Show, Generic)

--
-- JSON instances
--

instance ToJSON ExprCS where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON ExprCS

instance ToJSON SimpleCS where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON SimpleCS

instance ToJSON IfShortCS where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON IfShortCS

instance ToJSON WhileShortCS where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON WhileShortCS

instance ToJSON StatementShortCS where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON StatementShortCS

instance ToJSON StatementCS where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON StatementCS

instance ToJSON CoffeeScript where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON CoffeeScript

-- 
-- Generators
-- 

combineStringWithNumber :: String -> Int -> String
combineStringWithNumber s i = s ++ show i

arbitraryIdentifier :: Int -> Gen String
arbitraryIdentifier upperBound = elements $ map (combineStringWithNumber "a") [0 .. upperBound]
                                            
arbitraryConstant :: Gen Int
arbitraryConstant = elements $ [0 .. 10]

arbitrarySizedExpr :: Int -> Int -> Gen ExprCS
arbitrarySizedExpr upperBound n | n <= 0 = if upperBound == -1 
                                           then Const <$> arbitraryConstant
                                           else frequency [(1, Var <$> arbitraryIdentifier upperBound), (1, Const <$> arbitraryConstant)]
                                | otherwise = frequency ((if upperBound == -1 then [] else [(1, Var <$> arbitraryIdentifier upperBound)]) 
                                                ++ [(1, Const <$> arbitraryConstant)
                                                   ,(1, Plus <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2)) 
                                                   ,(1, Times <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2))
                                                   ,(1, Equal <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2))])
                                                             
instance Arbitrary ExprCS where
    arbitrary = sized (arbitrarySizedExpr 10)

    shrink n@(Plus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n@(Times e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n@(Equal e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n = [n]

arbitrarySizedSimpleCS :: Int -> Int -> Gen (SimpleCS, Int)
arbitrarySizedSimpleCS upperBound n | n <= 2 = do expr <- arbitrarySizedExpr upperBound n
                                                  return (Expr expr, upperBound)
                                    | otherwise = frequency [(1, (,upperBound+1) <$> (Assign <$> arbitraryIdentifier (upperBound + 1) <*> arbitrarySizedExpr upperBound (n - 1))),
                                                             (1, (,upperBound) <$> (Expr <$> arbitrarySizedExpr upperBound n))]

instance Arbitrary SimpleCS where
  arbitrary = fst <$> sized (arbitrarySizedSimpleCS 10)

arbitrarySizedIfShortCS :: Int -> Int -> Gen (IfShortCS, Int)
arbitrarySizedIfShortCS upperBound n | n <= 3 = do expr <- arbitrarySizedExpr upperBound n
                                                   (simple, _) <- arbitrarySizedSimpleCS upperBound n
                                                   return (IfSimple simple expr, upperBound)
                                     | otherwise = frequency [(1, (,upperBound) <$> (IfSimple <$> (fst <$> arbitrarySizedSimpleCS upperBound (n `div` 2)) <*> arbitrarySizedExpr upperBound (n `div` 2))),
                                                              (1, (,upperBound) <$> (IfComplex <$> (fst <$> arbitrarySizedIfShortCS upperBound (n `div` 2)) <*> arbitrarySizedExpr upperBound (n `div` 2)))]

instance Arbitrary IfShortCS where
  arbitrary = fst <$> sized (arbitrarySizedIfShortCS 10)

arbitrarySizedWhileShortCS :: Int -> Int -> Gen (WhileShortCS, Int)
arbitrarySizedWhileShortCS upperBound n | n <= 3 = do expr <- arbitrarySizedExpr upperBound n
                                                      (simple, _) <- arbitrarySizedSimpleCS upperBound n
                                                      return (WhileSimple simple expr, upperBound)
                                        | otherwise = frequency [(1, (,upperBound) <$> (WhileSimple <$> (fst <$> arbitrarySizedSimpleCS upperBound (n `div` 2)) <*> arbitrarySizedExpr upperBound (n `div` 2))),
                                                                 (1, (,upperBound) <$> (WhileComplex <$> (fst <$> arbitrarySizedWhileShortCS upperBound (n `div` 2)) <*> arbitrarySizedExpr upperBound (n `div` 2)))]

instance Arbitrary WhileShortCS where
  arbitrary = fst <$> sized (arbitrarySizedWhileShortCS 10)

-- data StatementCS = ShortStatementCS StatementShortCS | If ExprCS CoffeeScript | While ExprCS CoffeeScript | IfElse ExprCS CoffeeScript CoffeeScript 
--                     | IfThenElse ExprCS CoffeeScript CoffeeScript deriving (Read, Show, Generic)
-- data CoffeeScript = SimpleCS StatementCS | Complex CoffeeScript StatementCS deriving (Read, Show, Generic)

arbitrarySizedStatementShortCS :: Int -> Int -> Gen (StatementShortCS, Int)
arbitrarySizedStatementShortCS upperBound n | n <= 3 = first SimpleStatement <$> arbitrarySizedSimpleCS upperBound n
                                            | otherwise = frequency [(1, first SimpleStatement <$> arbitrarySizedSimpleCS upperBound n),
                                                                     (1, first SimpleIf <$> arbitrarySizedIfShortCS upperBound n),
                                                                     (1, first SimpleWhile <$> arbitrarySizedWhileShortCS upperBound n)]

instance Arbitrary StatementShortCS where
  arbitrary = fst <$> sized (arbitrarySizedStatementShortCS 10)

arbitrarySizedStatementCS :: Int -> Int -> Gen (StatementCS, Int)
arbitrarySizedStatementCS upperBound n | n <= 2 = first ShortStatementCS <$> arbitrarySizedStatementShortCS upperBound n
                                       | otherwise = frequency [(1, first ShortStatementCS <$> arbitrarySizedStatementShortCS upperBound n),
                                                                (1, (,upperBound) <$> (If <$> arbitrarySizedExpr upperBound (n `div` 2) <*> (fst <$> arbitrarySizedCoffeeScript upperBound (n `div` 2)))),
                                                                (1, (,upperBound) <$> (While <$> arbitrarySizedExpr upperBound (n `div` 2) <*> (fst <$> arbitrarySizedCoffeeScript upperBound (n `div` 2)))),
                                                                (1, (,upperBound) <$> (IfElse <$> arbitrarySizedExpr upperBound (n `div` 3) <*> (fst <$> arbitrarySizedCoffeeScript upperBound (n `div` 3))
                                                                 <*> (fst <$> arbitrarySizedCoffeeScript upperBound (n `div` 3)))),
                                                                (1, (,upperBound) <$> (IfThenElse <$> arbitrarySizedExpr upperBound (n `div` 3) <*> (fst <$> arbitrarySizedStatementShortCS upperBound (n `div` 3))
                                                                 <*> (fst <$> arbitrarySizedStatementShortCS upperBound (n `div` 3))))]

instance Arbitrary StatementCS where
  arbitrary = fst <$> sized (arbitrarySizedStatementCS 10)

arbitrarySizedCoffeeScript :: Int -> Int -> Gen (CoffeeScript, Int)
arbitrarySizedCoffeeScript upperBound n | n <= 2 = first SimpleCS <$> arbitrarySizedStatementCS upperBound n
                                        | otherwise = frequency [(1, first SimpleCS <$> arbitrarySizedStatementCS upperBound n),
                                                                 (1, arbitrarySizedCoffeeScript upperBound (n `div` 2) >>= \(cs, newBound) -> 
                                                                     first (ComplexCS cs) <$> arbitrarySizedStatementCS newBound (n `div` 2))]

instance Arbitrary CoffeeScript where
  arbitrary = fst <$> sized (arbitrarySizedCoffeeScript 10)
