{-# OPTIONS_GHC -Wall #-}

module LambdaTypeChecker where

import LambdaParser

import Data.Map (Map)
import qualified Data.Map as Map

import Data.Function

type Context = Map String Type

data TypeError = ExpectedFunction LambdaExpression Type
               | Mismatch LambdaExpression Type Type {- expression, got, expected -}
               | UndefinedVariable String 
               | NoEqualityForFunctions LambdaExpression 
               | ExpectedPair LambdaExpression Type

instance Show TypeError where
  show (ExpectedFunction e t) = "Expected a function type for the expression, " ++ show e ++ ", but this expression has type " ++ show t ++ "."
  show (Mismatch e t t') = "Expected type " ++ show t' ++ ", but got type " ++ show t ++ " for the expression, " ++ show e ++ "."
  show (UndefinedVariable var) = show var ++ " is an undefined variable name."
  show (NoEqualityForFunctions e) = "Function equality is not supported and the following expression contains a function type and was part of an equality expression, " ++ show e ++ "."
  show (ExpectedPair e t) = "Expected a pair type for the expression, " ++ show e ++ ", but this expression has type " ++ show t ++ "."

functionTypePresent :: Type -> Bool
functionTypePresent (TFun _ _) = True
functionTypePresent (TPair x y) = on (||) functionTypePresent x y
functionTypePresent _ = False

typeOf :: Context -> LambdaExpression -> Either TypeError Type
typeOf _ (Number _) = pure TInt
typeOf _ (Boolean _) = pure TBool
typeOf con (Variable x) = maybe (Left $ UndefinedVariable x) pure (Map.lookup x con)
typeOf con (Pair e e') = do x <- typeOf con e
                            y <- typeOf con e'
                            return $ TPair x y
typeOf con (Abstraction (x,t) expr) = TFun t <$> typeOf (Map.insert x t con) expr

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
                                                    TBool | y == x -> pure TBool
                                                          | otherwise -> Left $ Mismatch e' y x
                                                    TInt  | y == x -> pure TBool
                                                          | otherwise -> Left $ Mismatch e' y x
                                                    TPair _ _ | functionTypePresent x -> Left $ NoEqualityForFunctions e
                                                              | x == y -> pure TBool
                                                              | otherwise -> Left $ Mismatch e' y x
                                                    _ -> Left $ NoEqualityForFunctions e

typeOf con (UnaryOper Neg e) = do x <- typeOf con e
                                  case x of
                                        TInt -> pure TInt
                                        _ -> Left $ Mismatch e x TInt
typeOf con (UnaryOper Not e) = do x <- typeOf con e
                                  case x of
                                        TBool -> pure TBool
                                        _ -> Left $ Mismatch e x TBool
typeOf con (UnaryOper First e) = do x <- typeOf con e
                                    case x of
                                        TPair a _ -> pure a
                                        _ -> Left $ ExpectedPair e x
typeOf con (UnaryOper Second e) = do x <- typeOf con e
                                     case x of
                                        TPair _ b -> pure b
                                        _ -> Left $ ExpectedPair e x


typeOf con (If cond e e') = do x <- typeOf con cond
                               y <- typeOf con e
                               z <- typeOf con e'
                               case x of
                                TBool | y == z -> pure y
                                      | otherwise -> Left $ Mismatch e' z y
                                _ -> Left $ Mismatch cond x TBool

typeOf con (TypedExpression e t) = typeOf con e >>= \t' -> if t == t' then pure t else Left $ Mismatch e t' t

typeOf con (Let var e e') = typeOf con e >>= \x -> typeOf (Map.insert var x con) e'

typeOf con (LetRec (x,t) e e') = typeOf modifiedContext e >>= \t' -> if t' == t then typeOf modifiedContext e' else Left $ Mismatch e t' t
        where modifiedContext = Map.insert x t con