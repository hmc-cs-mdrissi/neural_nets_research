{-# LANGUAGE TupleSections, DeriveGeneric #-}

module ArbitraryCoffeeScriptTests where 

import Test.QuickCheck
import GHC.Generics
import Control.Arrow (first)
import Data.Aeson
import Prelude hiding (writeFile)

-- 
-- ArbitraryCoffeescriptTests.hs
-- A set of functions for generating arbitrary coffee script programs.
-- 

data ExprCS = Var String | Const Int | Plus ExprCS ExprCS | Times ExprCS ExprCS | Equal ExprCS ExprCS deriving Generic
data SimpleCS = Assign String ExprCS | Expr ExprCS deriving Generic
data IfShortCS = IfSimple SimpleCS ExprCS | IfComplex IfShortCS ExprCS deriving Generic
data WhileShortCS = WhileSimple SimpleCS ExprCS | WhileComplex WhileShortCS ExprCS deriving Generic
data StatementShortCS = SimpleStatement SimpleCS | SimpleIf IfShortCS | SimpleWhile WhileShortCS deriving Generic
data StatementCS = ShortStatementCS StatementShortCS | If ExprCS CoffeeScript | While ExprCS CoffeeScript | IfElse ExprCS CoffeeScript CoffeeScript 
                    | IfThenElse ExprCS StatementShortCS StatementShortCS deriving Generic
data CoffeeScript = SimpleCS StatementCS | ComplexCS CoffeeScript StatementCS deriving Generic

--
-- Show instances
--

showExpr :: Int -> ExprCS -> String
showExpr 0 (Equal x y) = showExpr 1 x ++ " == " ++ showExpr 1 y
showExpr 0 exp = showExpr 1 exp

showExpr 1 (Plus x y) = showExpr 1 x ++ " + " ++ showExpr 2 y
showExpr 1 exp = showExpr 2 exp

showExpr 2 (Times x y) = showExpr 2 x ++ " * " ++ showExpr 3 y
showExpr 2 exp = showExpr 3 exp

showExpr 3 (Var v) = v
showExpr 3 (Const n) = show n
showExpr _ exp = "(" ++ showExpr 0 exp ++ ")"

instance Show ExprCS where
    show = showExpr 0

instance Show SimpleCS where
    show (Assign v e) = v ++ " = " ++ show e
    show (Expr e) = show e

instance Show IfShortCS where
    show (IfSimple p e) = show p ++ " if " ++ show e
    show (IfComplex p e) = show p ++ " if " ++ show e

instance Show WhileShortCS where
    show (WhileSimple p e) = show p ++ " while " ++ show e
    show (WhileComplex p e) = show p ++ " while " ++ show e

instance Show StatementShortCS where
    show (SimpleStatement p) = show p
    show (SimpleIf p) = show p
    show (SimpleWhile p) = show p

showStatement :: Int -> StatementCS -> String
showStatement indentation_level (ShortStatementCS p) = replicate indentation_level '\t' ++ show p
showStatement indentation_level (If e p) = replicate indentation_level '\t' ++ "if " ++ show e ++ "\n" ++ showCoffeScript (indentation_level + 1) p
showStatement indentation_level (While e p) = replicate indentation_level '\t' ++ "while " ++ show e ++ "\n" ++ showCoffeScript (indentation_level + 1) p
showStatement indentation_level (IfElse e p1 p2) = replicate indentation_level '\t' ++ "if " ++ show e ++ "\n" ++ showCoffeScript (indentation_level + 1) p1 ++ "\n" ++
                                                   replicate indentation_level '\t' ++ "else\n" ++ showCoffeScript (indentation_level + 1) p2
showStatement indentation_level (IfThenElse e p1 p2) = replicate indentation_level '\t' ++ "if " ++ show e ++ " then " ++ show p1 ++ " else " ++ show p2

instance Show StatementCS where
    show = showStatement 0

showCoffeScript :: Int -> CoffeeScript -> String
showCoffeScript indentation_level (SimpleCS p) = showStatement indentation_level p
showCoffeScript indentation_level (ComplexCS p1 p2) = showCoffeScript indentation_level p1 ++ "\n" ++ showStatement indentation_level p2

instance Show CoffeeScript where
    show = showCoffeScript 0
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
max_int :: Int
max_int = 9


combineStringWithNumber :: String -> Int -> String
combineStringWithNumber s i = s ++ show i

arbitraryIdentifier :: Int -> Gen String
arbitraryIdentifier upperBound = elements $ map (combineStringWithNumber "a") [0 .. upperBound]
                                            
arbitraryConstant :: Gen Int
arbitraryConstant = elements $ [0 .. max_int]

arbitrarySizedExpr :: Int -> Int -> Gen ExprCS
arbitrarySizedExpr upperBound n | n <= 0 = if upperBound == -1 
                                           then Const <$> arbitraryConstant
                                           else frequency [(1, Var <$> arbitraryIdentifier upperBound), (1, Const <$> arbitraryConstant)]
                                | otherwise = frequency ((if upperBound == -1 then [] else [(1, Var <$> arbitraryIdentifier upperBound)]) 
                                                ++ [(1, Const <$> arbitraryConstant)
                                                   ,(2, Plus <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2)) 
                                                   ,(2, Times <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2))
                                                   ,(1, Equal <$> arbitrarySizedExpr upperBound ((n `div` 2) - 1) <*> arbitrarySizedExpr upperBound ((n `div` 2) - 2))])
                                                             
instance Arbitrary ExprCS where
    arbitrary = sized (arbitrarySizedExpr (-1))

    shrink n@(Plus e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n@(Times e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n@(Equal e1 e2) = shrink e1 ++ shrink e2 ++ [n]
    shrink n = [n]

arbitrarySizedSimpleCS :: Int -> Int -> Gen (SimpleCS, Int)
arbitrarySizedSimpleCS upperBound n | n <= 0 = do expr <- arbitrarySizedExpr upperBound n
                                                  return (Expr expr, upperBound)
                                    | otherwise = frequency ((if upperBound == -1 then [] else [(1, (,upperBound) <$> (Assign <$> arbitraryIdentifier upperBound <*> arbitrarySizedExpr upperBound (n - 1)))]) 
                                                  ++ [(3, (,upperBound+1) <$> (Assign (combineStringWithNumber "a" (upperBound + 1)) <$> arbitrarySizedExpr upperBound (n - 1))),
                                                      (1, (,upperBound) <$> (Expr <$> arbitrarySizedExpr upperBound n))])

instance Arbitrary SimpleCS where
  arbitrary = fst <$> sized (arbitrarySizedSimpleCS (-1))

arbitrarySizedIfShortCS :: Int -> Int -> Gen (IfShortCS, Int)
arbitrarySizedIfShortCS upperBound n | n <= 2 = do expr <- arbitrarySizedExpr upperBound n
                                                   (simple, _) <- arbitrarySizedSimpleCS upperBound n
                                                   return (IfSimple simple expr, upperBound)
                                     | otherwise = frequency [(1, (,upperBound) <$> (IfSimple <$> (fst <$> arbitrarySizedSimpleCS upperBound (n `div` 2)) <*> arbitrarySizedExpr upperBound (n `div` 2))),
                                                              (1, (,upperBound) <$> (IfComplex <$> (fst <$> arbitrarySizedIfShortCS upperBound (n `div` 2)) <*> arbitrarySizedExpr upperBound (n `div` 2)))]

instance Arbitrary IfShortCS where
  arbitrary = fst <$> sized (arbitrarySizedIfShortCS (-1))

arbitrarySizedWhileShortCS :: Int -> Int -> Gen (WhileShortCS, Int)
arbitrarySizedWhileShortCS upperBound n | n <= 2 = do expr <- arbitrarySizedExpr upperBound n
                                                      (simple, _) <- arbitrarySizedSimpleCS upperBound n
                                                      return (WhileSimple simple expr, upperBound)
                                        | otherwise = frequency [(1, (,upperBound) <$> (WhileSimple <$> (fst <$> arbitrarySizedSimpleCS upperBound (n `div` 2)) <*> arbitrarySizedExpr upperBound (n `div` 2))),
                                                                 (1, (,upperBound) <$> (WhileComplex <$> (fst <$> arbitrarySizedWhileShortCS upperBound (n `div` 2)) <*> arbitrarySizedExpr upperBound (n `div` 2)))]

instance Arbitrary WhileShortCS where
  arbitrary = fst <$> sized (arbitrarySizedWhileShortCS (-1))

arbitrarySizedStatementShortCS :: Int -> Int -> Gen (StatementShortCS, Int)
arbitrarySizedStatementShortCS upperBound n | n <= 3 = first SimpleStatement <$> arbitrarySizedSimpleCS upperBound n
                                            | otherwise = frequency [(2, first SimpleStatement <$> arbitrarySizedSimpleCS upperBound n),
                                                                     (1, first SimpleIf <$> arbitrarySizedIfShortCS upperBound n),
                                                                     (1, first SimpleWhile <$> arbitrarySizedWhileShortCS upperBound n)]

instance Arbitrary StatementShortCS where
  arbitrary = fst <$> sized (arbitrarySizedStatementShortCS (-1))

arbitrarySizedStatementCS :: Int -> Int -> Gen (StatementCS, Int)
arbitrarySizedStatementCS upperBound n | n <= 2 = first ShortStatementCS <$> arbitrarySizedStatementShortCS upperBound n
                                       | otherwise = frequency [(2, first ShortStatementCS <$> arbitrarySizedStatementShortCS upperBound n),
                                                                (1, (,upperBound) <$> (If <$> arbitrarySizedExpr upperBound (n `div` 2) <*> (fst <$> arbitrarySizedCoffeeScript upperBound (n `div` 2)))),
                                                                (1, (,upperBound) <$> (While <$> arbitrarySizedExpr upperBound (n `div` 2) <*> (fst <$> arbitrarySizedCoffeeScript upperBound (n `div` 2)))),
                                                                (1, (,upperBound) <$> (IfElse <$> arbitrarySizedExpr upperBound (n `div` 3) <*> (fst <$> arbitrarySizedCoffeeScript upperBound (n `div` 3))
                                                                 <*> (fst <$> arbitrarySizedCoffeeScript upperBound (n `div` 3)))),
                                                                (1, (,upperBound) <$> (IfThenElse <$> arbitrarySizedExpr upperBound (n `div` 3) <*> (fst <$> arbitrarySizedStatementShortCS upperBound (n `div` 3))
                                                                 <*> (fst <$> arbitrarySizedStatementShortCS upperBound (n `div` 3))))]

instance Arbitrary StatementCS where
  arbitrary = fst <$> sized (arbitrarySizedStatementCS (-1))

arbitrarySizedCoffeeScript :: Int -> Int -> Gen (CoffeeScript, Int)
arbitrarySizedCoffeeScript upperBound n | n <= 2 = first SimpleCS <$> arbitrarySizedStatementCS upperBound n
                                        | otherwise = frequency [(1, first SimpleCS <$> arbitrarySizedStatementCS upperBound n),
                                                                 (2, arbitrarySizedCoffeeScript upperBound (n `div` 2) >>= \(cs, newBound) -> 
                                                                     first (ComplexCS cs) <$> arbitrarySizedStatementCS newBound (n `div` 2))]

instance Arbitrary CoffeeScript where
  arbitrary = fst <$> sized (arbitrarySizedCoffeeScript (-1))
