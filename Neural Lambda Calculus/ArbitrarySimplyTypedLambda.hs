{-# LANGUAGE TupleSections #-}

module ArbitrarySimplyTypedLambda where

import SimplyTypedLambdaParser
import Test.QuickCheck
import Control.Applicative (liftA2)

-- 
-- ArbitrarySimplyTypedLambda.hs
-- A set of functions for generating arbitrary simply typed lambda calculus programs.
-- 

import Data.Map (Map, toList, member, insert)
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

arbitraryType :: Int -> Gen Type
arbitraryType n | n <= 1 = frequency [(1, return TInt), (1, return TBool), (1, return TIntList)]
                | otherwise = frequency [(1, arbitraryType 1), (1, TFun <$> arbitraryType 1 <*> arbitraryType (n - 1))]

arbitraryVariable :: Context -> Gen LambdaExpression
arbitraryVariable context = Variable <$> freshVariableGivenContext context

arbitraryVariableGivenContext :: Context -> Type -> Gen LambdaExpression
arbitraryVariableGivenContext context t = Variable <$> getArbitraryIdentifierFromContext context t

arbitraryAbstraction :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryAbstraction context (TFun domain range) n = do keyIdentifier <- freshVariableGivenContext context
                                                        lc <- arbitrarySizedSimplyTypedLambda (insert keyIdentifier domain context) range (n - 1)
                                                        return (Abstraction (keyIdentifier, domain) lc)
arbitraryAbstraction _ _ _ = error $ "You must generate a function type."

arbitraryNumber :: Gen LambdaExpression
arbitraryNumber = Number <$> arbitraryConstant

arbitraryBoolean :: Gen LambdaExpression
arbitraryBoolean = Boolean <$> elements [True, False]

arbitraryBinaryOpArithmetic :: Gen BinaryOp
arbitraryBinaryOpArithmetic = frequency [(1, return Plus), (1, return Minus), (1, return Times), (1, return Divide)]

arbitraryBinaryOpLogical :: Gen BinaryOp
arbitraryBinaryOpLogical = frequency [(1, return And), (1, return Or)]

arbitraryIf :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryIf context t n = do cond <- arbitrarySizedSimplyTypedLambda context TBool (n `div` 3)
                             ifcase <- arbitrarySizedSimplyTypedLambda context t (n `div` 3)
                             elsecase <- arbitrarySizedSimplyTypedLambda context t (n `div` 3)
                             return (If cond ifcase elsecase)

arbitraryNil ::  Gen LambdaExpression
arbitraryNil = return Nil

arbitraryCons ::  Context -> Int -> Gen LambdaExpression
arbitraryCons context n | n <= 1 = do number <- arbitraryNumber
                                      return (Cons number Nil)
                        | otherwise = do number <- arbitrarySizedSimplyTypedLambda context TInt (n `div` 2) 
                                         sublist <- arbitrarySizedSimplyTypedLambda context TIntList (n `div` 2)
                                         return (Cons number sublist)

arbitraryFunction :: Gen Type
arbitraryFunction = TFun <$> (arbitraryType 1) <*> (arbitraryType 1)

arbitraryUnaryOper :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryUnaryOper context TInt n = UnaryOper Neg <$> arbitrarySizedSimplyTypedLambda context TInt (n - 1)
arbitraryUnaryOper context TBool n = UnaryOper Not <$> arbitrarySizedSimplyTypedLambda context TBool (n - 1)
arbitraryUnaryOper context _ n = error $ "You can't use arbitrary unary op to generate any other types."

arbitraryMatch :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryMatch context t n = do ls <- arbitrarySizedSimplyTypedLambda context TIntList (n `div` 3)
                                nilcase <- arbitrarySizedSimplyTypedLambda context t (n `div` 3)
                                conscase <- arbitrarySizedSimplyTypedLambda context t (n `div` 3)
                                return $ Match ls nilcase conscase

arbitraryLet :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryLet context t n = do keyIdentifier <- freshVariableGivenContext context
                              valueType <- arbitraryType 3
                              assignee <- arbitrarySizedSimplyTypedLambda context valueType (n `div` 2)
                              body <- arbitrarySizedSimplyTypedLambda (insert keyIdentifier valueType context) t (n `div` 2)
                              return $ Let keyIdentifier assignee body

arbitraryLetRec :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryLetRec context t n = do keyIdentifier <- freshVariableGivenContext context
                                 domain <- arbitraryType 1
                                 range <- arbitraryType 1
                                 let valueType = TFun domain range
                                 assignee <- arbitrarySizedSimplyTypedLambda (insert keyIdentifier valueType context) valueType (n `div` 2)
                                 body <- arbitrarySizedSimplyTypedLambda (insert keyIdentifier valueType context) t (n `div` 2)
                                 return $ LetRec (keyIdentifier, valueType) assignee body

-- Note, we do not generate function types through binary operations
arbitraryBinaryOper :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryBinaryOper context TInt n = do domainType <- arbitraryType 1
                                        smallerFunction <- arbitrarySizedSimplyTypedLambda context (TFun domainType TInt) (n `div` 2)
                                        argument <- arbitrarySizedSimplyTypedLambda context domainType (n `div` 2)
                                        frequency[(4, BinaryOper <$> smallerLambdaExpression <*> arbitraryBinaryOpArithmetic <*> smallerLambdaExpression)
                                                 ,(1, return $ BinaryOper smallerFunction Application argument)] 
                                        where smallerLambdaExpression = arbitrarySizedSimplyTypedLambda context TInt (n `div` 2)
arbitraryBinaryOper context TBool n = do domainType <- arbitraryType 1
                                         smallerFunction <- arbitrarySizedSimplyTypedLambda context (TFun domainType TBool) (n `div` 2)
                                         argument <- arbitrarySizedSimplyTypedLambda context domainType (n `div` 2)
                                         frequency [(1, BinaryOper <$> smallerIntExpression <*> return Equal <*> smallerIntExpression)
                                                   ,(1, BinaryOper <$> smallerIntExpression <*> return Less <*> smallerIntExpression)
                                                   ,(2, BinaryOper <$> smallerBoolExpression <*> arbitraryBinaryOpLogical <*> smallerBoolExpression)
                                                   ,(1, BinaryOper <$> smallerIntList <*> return Equal <*> smallerIntList)
                                                   ,(1, return $ BinaryOper smallerFunction Application argument)]
                                         where smallerIntExpression = arbitrarySizedSimplyTypedLambda context TInt (n `div` 2)
                                               smallerBoolExpression = arbitrarySizedSimplyTypedLambda context TBool (n `div` 2)
                                               smallerIntList = arbitrarySizedSimplyTypedLambda context TIntList (n `div` 2)
arbitraryBinaryOper context _ n = error $ "Binary operation not allowed."

--data Type = TInt | TBool | TIntList | TFun Type Type | TFake deriving (Eq, Generic)

--data LambdaExpression = 
--    Variable String | -- 1
--    Abstraction (String, Type) LambdaExpression | -- 1
--    Number Integer | Boolean Bool | Nil | -- Number : 1, Boolean : 11, Nil : 21
--    If LambdaExpression LambdaExpression LambdaExpression | -- 11
--    Cons LambdaExpression LambdaExpression | -- 21
--    Match LambdaExpression LambdaExpression LambdaExpression | -- 21
--    UnaryOper UnaryOp LambdaExpression | -- 1
--    BinaryOper LambdaExpression BinaryOp LambdaExpression | -- 1 (except no application until 31)
--    Let String LambdaExpression LambdaExpression | -- 31
--    LetRec (String, Type) LambdaExpression LambdaExpression deriving (Eq, Generic) -- 41

--data UnaryOp = Neg | Not deriving (Eq, Generic)
--data BinaryOp = Plus | Minus | Times | Divide | And | Or | Equal | Less | Application deriving (Eq, Generic)

testContext :: Context
testContext = Map.fromList [("myInt0", TInt), ("myInt1", TInt), ("myBool0", TBool), ("myBool1", TBool), ("myIntList", TIntList)]

-- 
-- General Generator
-- 

arbitrarySizedSimplyTypedLambda :: Context -> Type -> Int -> Gen LambdaExpression
arbitrarySizedSimplyTypedLambda context TInt n | n <= 1 = arbitraryNumber
                                               | otherwise = frequency [(1, arbitraryNumber)
                                                                       ,(1, arbitraryIf context TInt n)
                                                                       --,(1, arbitraryMatch context TInt n)
                                                                       ,(1, arbitraryUnaryOper context TInt n)
                                                                       ,(1, arbitraryBinaryOper context TInt n)
                                                                       ,(1, arbitraryLet context TInt n)
                                                                       ,(1, arbitraryLetRec context TInt n)]
arbitrarySizedSimplyTypedLambda context TBool n | n <= 1 = arbitraryBoolean 
                                                | otherwise = frequency [(1, arbitraryIf context TBool n)
                                                                        --,(1, arbitraryMatch context TBool n)
                                                                        ,(1, arbitraryUnaryOper context TBool n)
                                                                        ,(1, arbitraryBinaryOper context TBool n)
                                                                        ,(1, arbitraryLet context TBool n)
                                                                        ,(1, arbitraryLetRec context TBool n)]
arbitrarySizedSimplyTypedLambda context TIntList n | n <= 1 = arbitraryNil
                                                   | otherwise = frequency [(1, arbitraryIf context TIntList n)
                                                                           --,(1, arbitraryMatch context TIntList n)
                                                                           ,(2, arbitraryCons context n)
                                                                           ,(1, arbitraryBinaryOper context TIntList n)
                                                                           ,(1, arbitraryLet context TIntList n)
                                                                           ,(1, arbitraryLetRec context TIntList n)]
arbitrarySizedSimplyTypedLambda context functionType@(TFun domain range) n = frequency [(1, arbitraryAbstraction context functionType n)]
arbitrarySizedSimplyTypedLambda context t n = error $ "Arbitrary lc of type " ++ (show t) ++ " not supported."
