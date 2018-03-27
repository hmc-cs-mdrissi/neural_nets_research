module ArbitrarySimplyTypedLambda where

import SimplyTypedLambdaParser
import Test.QuickCheck

-- 
-- ArbitrarySimplyTypedLambda.hs
-- A set of functions for generating arbitrary simply typed lambda calculus programs.
-- 

import Data.Map (Map)
import qualified Data.Map as Map

type Context = Map String Type

combineStringWithNumber :: String -> Int -> String
combineStringWithNumber s i = s ++ show i

arbitraryIdentifier :: Int -> Gen String
arbitraryIdentifier upperBound = do identifier <- elements $ map (combineStringWithNumber "a") [1 .. upperBound]
                                    return identifier

arbitraryConstant :: Gen Integer
arbitraryConstant = do firstInt <- elements $ [1 .. 10]
                       return firstInt

--data LambdaExpression = 
--    Variable String |
--    Abstraction (String, Type) LambdaExpression | 
--    Number Integer | Boolean Bool | Nil |
--    If LambdaExpression LambdaExpression LambdaExpression |
--    Cons LambdaExpression LambdaExpression | 
--    Match LambdaExpression LambdaExpression LambdaExpression |
--    UnaryOper UnaryOp LambdaExpression |
--    BinaryOper LambdaExpression BinaryOp LambdaExpression |
--    Let String LambdaExpression LambdaExpression |
--    LetRec (String, Type) LambdaExpression LambdaExpression deriving (Eq, Generic)

-- data UnaryOp = Neg | Not deriving (Eq, Generic)
-- data BinaryOp = Plus | Minus | Times | Divide | And | Or | Equal | Less | Application deriving (Eq, Generic)

-- TFake is a type for evaluation purposes only when types are not examined.
-- data Type = TInt | TBool | TIntList | TFun Type Type | TFake deriving (Eq, Generic)


-- List of fully handled cases:
-- Number
-- Boolean (0.5)
-- Type

arbitraryType :: Int -> Gen Type
arbitraryType n | n <= 0 = frequency [(1, return TInt), (1, return TBool), (1, return TIntList), (1, return TFake)]
                | otherwise = TFun <$> smallerType <*> smallerType where smallerType = arbitraryType (n `div` 2)

arbitraryNumber :: Gen LambdaExpression
arbitraryNumber = Number <$> arbitraryConstant

arbitraryBinaryOpArithmetic :: Gen BinaryOp
arbitraryBinaryOpArithmetic = frequency [(1, return Plus), (1, return Minus), (1, return Times), (1, return Divide)]

-- < fix
arbitraryBinaryOpLogical :: Gen BinaryOp
arbitraryBinaryOpLogical = frequency [(1, return And), (1, return Or), (1, return Equal)]

arbitraryBoolean :: Int -> Context -> Gen LambdaExpression
arbitraryBoolean n context | n <= 0 = Boolean <$> (elements [True, False])
                           | otherwise = frequency [(1, BinaryOper <$> smallerBoolean <*> arbitraryBinaryOpLogical <*> smallerBoolean)]
                                         where smallerBoolean = arbitraryBoolean (n - 1) context

--arbitraryNumberList :: Int -> Context -> Gen LambdaExpression
--arbitraryNumberList n context = undefined

--arbitraryFunction :: Int -> Context -> Type -> Type -> Gen LambdaExpression
--arbitraryFunction n context type1 type2 = undefined

--arbitraryLambda :: Int -> Context -> Gen LambdaExpression
--arbitraryLambda n context = undefined

---- Return the variables in the context that 
--findType :: Type -> Context -> [String]
--findType type1 context = undefined 
