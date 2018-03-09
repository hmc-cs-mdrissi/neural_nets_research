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

arbitraryIdentifier :: Gen String
arbitraryIdentifier = do firstChar <- elements $ ['a' .. 'z']
                         return [firstChar]

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

--data UnaryOp = Neg | Not deriving (Eq, Generic)
--data BinaryOp = Plus | Minus | Times | Divide | And | Or | Equal | Less | Application deriving (Eq, Generic)

-- TFake is a type for evaluation purposes only when types are not examined.
-- data Type = TInt | TBool | TIntList | TFun Type Type | TFake deriving (Eq, Generic)

arbitraryType :: Int -> Gen Type
arbitraryType n = undefined

arbitraryNumber :: Int -> Context -> Gen LambdaExpression
arbitraryNumber n context = undefined

arbitraryBoolean :: Int -> Context -> Gen LambdaExpression
arbitraryBoolean n context = frequency [(1, BinaryOper <$> smallerArbitraryBoolean <*> And <*> smallerArbitraryBoolean),
                                        (1, BinaryOper <$> smallerArbitraryBoolean <*> Or <*> smallerArbitraryBoolean)]
                             where smallerArbitraryBoolean (n - 1) context 

arbitraryNumberList :: Int -> Context -> Gen LambdaExpression
arbitraryNumberList n context = undefined

arbitraryFunction :: Int -> Context -> Type -> Type -> Gen LambdaExpression
arbitraryFunction n context type1 type2 = undefined

arbitraryLambda :: Int -> Context -> Gen LambdaExpression
arbitraryLambda n context = undefined


-- Return the variables in the context that 
findType :: Type -> Context -> [String]
findType type1 context = undefined
