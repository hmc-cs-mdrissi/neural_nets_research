{-# OPTIONS_GHC -Wall #-}

module SimplyTypedLambdaTypeChecker where

import SimplyTypedLambdaParser

import Data.Map (Map)
import qualified Data.Map as Map

type Context = Map String Type

data TypeError = ExpectedFunction LambdaExpression Type
               | Mismatch LambdaExpression Type Type {- expression, got, expected -}
               | UndefinedVariable String 
               | NoEqualityForFunctions LambdaExpression 

instance Show TypeError where
  show (ExpectedFunction e t) = "Expected a function type for the expression, " ++ show e ++ ", but this expression has type " ++ show t ++ "."
  show (Mismatch e t t') = "Expected type " ++ show t' ++ ", but got type " ++ show t ++ " for the expression, " ++ show e ++ "."
  show (UndefinedVariable var) = show var ++ " is an undefined variable name."
  show (NoEqualityForFunctions e) = "Function equality is not supported and the following expression contains a function type and was part of an equality expression, " ++ show e ++ "."

typeOf :: Context -> LambdaExpression -> Either TypeError Type
typeOf _ (Number _) = pure TInt
typeOf _ (Boolean _) = pure TBool
typeOf _ Nil = pure TIntList
typeOf con (Variable x) = maybe (Left $ UndefinedVariable x) pure (Map.lookup x con)
typeOf con (Abstraction (x, t) expr) = TFun t <$> typeOf (Map.insert x t con) expr
typeOf con (Cons h t) = do x <- typeOf con h
                           y <- typeOf con t
                           case x of
                              TInt | y == TIntList -> pure TIntList
                                   | otherwise -> Left $ Mismatch t y TIntList
                              _ -> Left $ Mismatch h x TInt
typeOf con (BinaryOper e op e') | op `elem` [Plus, Minus, Times, Divide] = do x <- typeOf con e 
                                                                              y <- typeOf con e'
                                                                              case x of
                                                                                    TInt | y == x -> pure TInt
                                                                                         | otherwise -> Left $ Mismatch e' y x
                                                                                    _ -> Left $ Mismatch e x TInt
                                | op `elem` [And, Or] = do x <- typeOf con e 
                                                           y <- typeOf con e'
                                                           case x of
                                                                TBool | y == x -> pure TBool
                                                                      | otherwise -> Left $ Mismatch e' y x
                                                                _ -> Left $ Mismatch e x TBool
                                | op == Less = do x <- typeOf con e 
                                                  y <- typeOf con e'
                                                  case x of
                                                        TInt | y == x -> pure TBool
                                                             | otherwise -> Left $ Mismatch e' y x
                                                        _ -> Left $ Mismatch e x TInt
                                | op == Application = do t <- typeOf con e
                                                         t' <- typeOf con e'
                                                         case t of
                                                            TFun t1 t2 | t1 == t' -> pure $ t2
                                                                       | otherwise -> Left $ Mismatch e' t' t1
                                                            _ -> Left $ ExpectedFunction e t
                                | otherwise = do x <- typeOf con e
                                                 y <- typeOf con e'
                                                 case x of
                                                      TFun _ _ -> Left $ NoEqualityForFunctions e
                                                      _ | y == x -> pure x
                                                        | otherwise -> Left $ Mismatch e' y x
typeOf con (UnaryOper Neg e) = do x <- typeOf con e
                                  case x of
                                        TInt -> pure TInt
                                        _ -> Left $ Mismatch e x TInt
typeOf con (UnaryOper Not e) = do x <- typeOf con e
                                  case x of
                                        TBool -> pure TBool
                                        _ -> Left $ Mismatch e x TBool
typeOf con (If cond e e') = do x <- typeOf con cond
                               y <- typeOf con e
                               z <- typeOf con e'
                               case x of
                                TBool | y == z -> pure y
                                      | otherwise -> Left $ Mismatch e' z y
                                _ -> Left $ Mismatch cond x TBool
typeOf con (Let var e e') = typeOf con e >>= \x -> typeOf (Map.insert var x con) e'
typeOf con (LetRec (x,t) e e') = typeOf modifiedContext e >>= \t' -> if t' == t then typeOf modifiedContext e' else Left $ Mismatch e t' t
        where modifiedContext = Map.insert x t con
typeOf con (Match ls e e') = do x <- typeOf con ls
                                y <- typeOf con e
                                z <- typeOf con e'
                                case x of
                                    TIntList -> case z of
                                                    TFun _ (TFun _ y') | y == y' ->  pure y
                                                                       | otherwise -> Left $ Mismatch e' y' y
                                                    _ -> Left $ ExpectedFunction e' z
                                    _ -> Left $ Mismatch ls x TIntList