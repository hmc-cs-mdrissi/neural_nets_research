{-# LANGUAGE TupleSections #-}

module ArbitrarySimplyTypedLambda where

import SimplyTypedLambdaParser
import Test.QuickCheck
import Control.Applicative (liftA2)

-- 
-- ArbitrarySimplyTypedLambda.hs
-- A set of functions for generating arbitrary simply typed lambda calculus programs.
-- 

import Data.Map (Map, toList, member)
import qualified Data.Map as Map

type Context = Map String Type

isTypeVariablePair :: Type -> (Type, b) -> Bool
isTypeVariablePair typ (typ', b) = if typ == typ' then True else False

freshVariableGivenContext :: Context -> Gen String
freshVariableGivenContext context = return $ freshVariableAux 1 context
                                    where freshVariableAux n context = if ("a" ++ show n) `member` context then freshVariableAux (n + 1) context else "a" ++ show n

getArbitraryIdentifierFromContext :: Context -> Type -> Gen String
getArbitraryIdentifierFromContext context typ = elements $ map fst $ filter ((==typ) . snd) $ (toList context)

arbitraryConstant :: Gen Integer
arbitraryConstant = do firstInt <- elements $ [0 .. 10]
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
arbitraryType n | n <= 0 = frequency [(1, return TInt), (1, return TBool), (1, return TIntList)]
                | otherwise = TFun <$> smallerType <*> smallerType where smallerType = arbitraryType (n `div` 2)

arbitraryBinaryOpArithmetic :: Gen BinaryOp
arbitraryBinaryOpArithmetic = frequency [(1, return Plus), (1, return Minus), (1, return Times), (1, return Divide)]

arbitraryBinaryOpLogical :: Gen BinaryOp
arbitraryBinaryOpLogical = frequency [(1, return And), (1, return Or)]

keepContext :: Monad m => Context -> m a -> m (a, Context)
keepContext context ma = do a <- ma
                            return (a, context)

arbitraryVariable :: Context -> Gen (LambdaExpression, Context)
arbitraryVariable context = keepContext context (Variable <$> freshVariableGivenContext context)

arbitraryNumber :: Context -> Gen (LambdaExpression, Context)
arbitraryNumber context = keepContext context (Number <$> arbitraryConstant)

arbitraryBoolean :: Context -> Gen (LambdaExpression, Context)
arbitraryBoolean context = keepContext context (Boolean <$> (elements [True, False])) 

--arbitraryNumberList :: Int -> Context -> Gen LambdaExpression
--arbitraryNumberList n context = undefined

--arbitraryFunction :: Int -> Context -> Type -> Type -> Gen LambdaExpression
--arbitraryFunction n context type1 type2 = undefined

--arbitraryLambda :: Int -> Context -> Gen LambdaExpression
--arbitraryLambda n context = undefined

