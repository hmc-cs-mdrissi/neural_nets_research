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

-- 
-- Difficulty
-- 
data Difficulty = Debug | Easy | Medium | Hard | VeryHard deriving (Read, Show)

-- 
-- Context Query, Modify, Generate
-- 

type Context = Map String Type

isTypeVariablePair :: Type -> (Type, b) -> Bool
isTypeVariablePair typ (typ', b) = if typ == typ' then True else False

freshVariableGivenContext :: Context -> Gen String
freshVariableGivenContext context = return $ freshVariableAux 0 context
                                    where freshVariableAux n context = if ("a" ++ show n) `member` context then freshVariableAux (n + 1) context else "a" ++ show n

getArbitraryIdentifierFromContext :: Context -> Type -> Gen String
getArbitraryIdentifierFromContext context typ = elements $ map fst $ filter ((==typ) . snd) $ (toList context)

isIdentifierTypeInContext :: Context -> Type -> Bool
isIdentifierTypeInContext context typ = let filteredIdentifiers = filter ((==typ) . snd) $ (toList context) in
                                            if (null filteredIdentifiers) then False else True

arbitraryVariableGivenContext :: Context -> Type -> Gen LambdaExpression
arbitraryVariableGivenContext context t = Variable <$> getArbitraryIdentifierFromContext context t

-- 
-- Grammar Generation
-- 

arbitraryConstant :: Gen Integer
arbitraryConstant = do firstInt <- elements $ [0 .. 10]
                       return firstInt

arbitraryType :: Int -> Gen Type
arbitraryType n | n <= 1 = frequency [(1, return TInt), (1, return TBool), (1, return TIntList)]
                | otherwise = frequency [(1, arbitraryType 1), (1, TFun <$> arbitraryType 1 <*> arbitraryType (n - 1))]

arbitraryAbstraction :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryAbstraction context (TFun domain range) n = do keyIdentifier <- freshVariableGivenContext context
                                                        lc <- arbitrarySizedSimplyTypedLambda (insert keyIdentifier domain context) range (n - 2)
                                                        return (Abstraction (keyIdentifier, domain) lc)
arbitraryAbstraction _ _ _ = error $ "You must generate a function type."

arbitraryInt :: Context -> Gen LambdaExpression
arbitraryInt context = if isIdentifierTypeInContext context TInt 
                       then frequency [(3, Variable <$> getArbitraryIdentifierFromContext context TInt)
                                      ,(1, Number <$> arbitraryConstant)]
                       else Number <$> arbitraryConstant

arbitraryBoolean :: Context -> Gen LambdaExpression
arbitraryBoolean context = if isIdentifierTypeInContext context TBool
                           then frequency [(3, Variable <$> getArbitraryIdentifierFromContext context TBool)
                                          ,(1, Boolean <$> elements [True, False])]
                           else Boolean <$> elements [True, False]

arbitraryBinaryOpArithmetic :: Gen BinaryOp
arbitraryBinaryOpArithmetic = frequency [(1, return Plus), (1, return Minus), (1, return Times), (0, return Divide)]

arbitraryBinaryOpLogical :: Gen BinaryOp
arbitraryBinaryOpLogical = frequency [(1, return And), (1, return Or)]

arbitraryIf :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryIf context t n | n <= 1 = do smallCond <- arbitrarySizedSimplyTypedLambda context TBool 1
                                      smallIf <- arbitrarySizedSimplyTypedLambda context t 1
                                      smallElse <- arbitrarySizedSimplyTypedLambda context t 1
                                      return (If smallCond smallIf smallElse)
                        | otherwise = do cond <- arbitrarySizedSimplyTypedLambda context TBool (n `div` 3)
                                         ifcase <- arbitrarySizedSimplyTypedLambda context t (n `div` 3)
                                         elsecase <- arbitrarySizedSimplyTypedLambda context t (n `div` 3)
                                         return (If cond ifcase elsecase)

arbitraryNil ::  Gen LambdaExpression
arbitraryNil = return Nil

arbitraryCons ::  Context -> Int -> Gen LambdaExpression
arbitraryCons context n | n <= 3 = do number <- arbitraryInt context
                                      return (Cons number Nil)
                        | otherwise = do number <- arbitrarySizedSimplyTypedLambda context TInt (n `div` 2) 
                                         sublist <- arbitrarySizedSimplyTypedLambda context TIntList (n `div` 2)
                                         return (Cons number sublist)

arbitraryFunction :: Gen Type
arbitraryFunction = TFun <$> (arbitraryType 1) <*> (arbitraryType 1)

arbitraryUnaryOper :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryUnaryOper context TInt n | n <= 2 = UnaryOper Neg <$> arbitrarySizedSimplyTypedLambda context TInt 1
                                  | otherwise = UnaryOper Neg <$> arbitrarySizedSimplyTypedLambda context TInt (n - 1)
arbitraryUnaryOper context TBool n | n <= 2 = UnaryOper Not <$> arbitrarySizedSimplyTypedLambda context TBool 1
                                   | otherwise = UnaryOper Neg <$> arbitrarySizedSimplyTypedLambda context TBool (n - 1)
arbitraryUnaryOper context _ n = error $ "You can't use arbitrary unary op to generate any other types."

arbitraryMatch :: Context -> Type -> Int -> Gen LambdaExpression
arbitraryMatch context t n | n <= 6 = do baseList <- arbitrarySizedSimplyTypedLambda context TIntList 3
                                         baseNil <- arbitrarySizedSimplyTypedLambda context t 1
                                         baseCons <- arbitrarySizedSimplyTypedLambda (insert "t" TIntList (insert "h" TInt context)) t 1
                                         return $ Match baseList baseNil (Abstraction ("h", TInt) (Abstraction ("t", TIntList) baseCons))
                           | otherwise = do ls <- arbitrarySizedSimplyTypedLambda context TIntList (n `div` 3)
                                            nilcase <- arbitrarySizedSimplyTypedLambda context t (n `div` 3)
                                            conscase <- arbitrarySizedSimplyTypedLambda (insert "t" TIntList (insert "h" TInt context)) t (n `div` 3)
                                            return $ Match ls nilcase (Abstraction ("h", TInt) (Abstraction ("t", TIntList) conscase))

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
arbitraryBinaryOper context TInt n | n <= 6 = do baseDomainType <- arbitraryType 1
                                                 baseFunction <- arbitrarySizedSimplyTypedLambda context (TFun baseDomainType TInt) 1
                                                 baseArgument <- arbitrarySizedSimplyTypedLambda context baseDomainType 1
                                                 frequency [(4, BinaryOper <$> (arbitrarySizedSimplyTypedLambda context TInt 1) <*> arbitraryBinaryOpArithmetic <*> (arbitrarySizedSimplyTypedLambda context TInt 1))
                                                           ,(1, return $ BinaryOper baseFunction Application baseArgument)]
                                   | otherwise = do domainType <- arbitraryType 1
                                                    smallerFunction <- arbitrarySizedSimplyTypedLambda context (TFun domainType TInt) (n `div` 2)
                                                    argument <- arbitrarySizedSimplyTypedLambda context domainType (n `div` 2)
                                                    frequency[(4, BinaryOper <$> (arbitrarySizedSimplyTypedLambda context TInt (n `div` 2)) <*> arbitraryBinaryOpArithmetic <*> (arbitrarySizedSimplyTypedLambda context TInt (n `div` 2)))
                                                             ,(1, return $ BinaryOper smallerFunction Application argument)] 
arbitraryBinaryOper context TBool n | n <= 6 = do domainType <- arbitraryType 1
                                                  smallerFunction <- arbitrarySizedSimplyTypedLambda context (TFun domainType TBool) 1
                                                  argument <- arbitrarySizedSimplyTypedLambda context domainType 1
                                                  frequency [(1, BinaryOper <$> (arbitrarySizedSimplyTypedLambda context TInt 1) <*> return Equal <*> (arbitrarySizedSimplyTypedLambda context TInt 1))
                                                            ,(1, BinaryOper <$> (arbitrarySizedSimplyTypedLambda context TInt 1) <*> return Less <*> (arbitrarySizedSimplyTypedLambda context TInt 1))
                                                            ,(2, BinaryOper <$> (arbitrarySizedSimplyTypedLambda context TBool 1) <*> arbitraryBinaryOpLogical <*> (arbitrarySizedSimplyTypedLambda context TBool 1))
                                                            ,(1, BinaryOper <$> (arbitrarySizedSimplyTypedLambda context TIntList 1) <*> return Equal <*> (arbitrarySizedSimplyTypedLambda context TIntList 1))
                                                            ,(1, return $ BinaryOper smallerFunction Application argument)]
                                    | otherwise = do domainType <- arbitraryType 1
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
-- add pulling from context to each to the types below
arbitrarySizedSimplyTypedLambda context TInt n | n <= 1 = arbitraryInt context
                                               | otherwise = frequency [(1, arbitraryInt context)
                                                                       ,(1, arbitraryIf context TInt n)
                                                                       ,(1, arbitraryMatch context TInt n)
                                                                       ,(1, arbitraryUnaryOper context TInt n)
                                                                       ,(1, arbitraryBinaryOper context TInt n)
                                                                       ,(1, arbitraryLet context TInt n)
                                                                       ,(1, arbitraryLetRec context TInt n)]
arbitrarySizedSimplyTypedLambda context TBool n | n <= 1 = arbitraryBoolean context
                                                | otherwise = frequency [(1, arbitraryBoolean context)
                                                                        ,(1, arbitraryIf context TBool n)
                                                                        ,(1, arbitraryMatch context TBool n)
                                                                        ,(1, arbitraryUnaryOper context TBool n)
                                                                        ,(1, arbitraryBinaryOper context TBool n)
                                                                        ,(1, arbitraryLet context TBool n)
                                                                        ,(1, arbitraryLetRec context TBool n)]
arbitrarySizedSimplyTypedLambda context TIntList n | n <= 1 = arbitraryNil
                                                   | otherwise = frequency [(1, arbitraryNil)
                                                                           ,(1, arbitraryIf context TIntList n)
                                                                           ,(1, arbitraryMatch context TIntList n)
                                                                           ,(2, arbitraryCons context n)
                                                                           ,(1, arbitraryLet context TIntList n)
                                                                           ,(1, arbitraryLetRec context TIntList n)]
arbitrarySizedSimplyTypedLambda context functionType@(TFun domain range) n = frequency [(1, arbitraryAbstraction context functionType n)]
arbitrarySizedSimplyTypedLambda context t n = error $ "Arbitrary lc of type " ++ (show t) ++ " not supported."

generateLambdaExpressionWithTermLength :: Context -> Int -> Gen LambdaExpression
generateLambdaExpressionWithTermLength context n = frequency [(1, arbitrarySizedSimplyTypedLambda context (TFun TInt TInt) n)
                                                             ,(0, arbitrarySizedSimplyTypedLambda context TInt n)
                                                             ,(0, arbitrarySizedSimplyTypedLambda context TBool n)
                                                             ,(0, arbitrarySizedSimplyTypedLambda context TIntList n)]

-- TODO: Currently does not generate top-level abstractions
arbitrarySizedSimplyTypedLambdaWithDifficulty :: Difficulty -> Gen LambdaExpression
arbitrarySizedSimplyTypedLambdaWithDifficulty Easy = generateLambdaExpressionWithTermLength Map.empty 5
arbitrarySizedSimplyTypedLambdaWithDifficulty Medium = generateLambdaExpressionWithTermLength Map.empty 8
arbitrarySizedSimplyTypedLambdaWithDifficulty Hard = generateLambdaExpressionWithTermLength Map.empty 10
arbitrarySizedSimplyTypedLambdaWithDifficulty VeryHard = generateLambdaExpressionWithTermLength Map.empty 15







