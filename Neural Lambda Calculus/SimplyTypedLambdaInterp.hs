{-# OPTIONS_GHC -Wall #-}

module SimplyTypedLambdaInterp where

import SimplyTypedLambdaParser
import SimplyTypedLambdaTypeChecker

import Data.Function
import Data.Set (Set)
import qualified Data.Set as Set
import qualified Data.Map as Map

data EvaluationError = UndefinedVariableE String | TypeMismatchE Type LambdaExpression | ExpectedFunctionE LambdaExpression 
                                                 | DivideByZeroE LambdaExpression
                                                 | NoEqualityForFunctionsE LambdaExpression | RecursiveDefinitionLet LambdaExpression
                                                 | RecursiveDefinitionNonFunction LambdaExpression
                                                 | TypeError TypeError LambdaExpression

instance Show EvaluationError where
    show (UndefinedVariableE var) = "Evaluation Error - Undefined variable " ++ var ++ "."
    show (TypeMismatchE t e) = "Evaluation Error - Expected type: " ++ show t ++ ", but got " ++ show e ++ "."
    show (ExpectedFunctionE e) = "Evaluation Error - This expression, " ++ show e ++ ", was attempted to be applied to an argument, but it is not a function type."
    show (DivideByZeroE e) = "Evaluation Error - An attempt to divide by zero occurred in this expression, " ++ show e ++ "."
    show (NoEqualityForFunctionsE e) = "Evaluation Error - Trying to do an equality with this expression is not possible, " ++ show e ++ ", because it is a function."
    show (RecursiveDefinitionLet e) = "Evaluation Error - Trying to do a recursive definition in this let expression, " ++ show e ++ ", is not allowed."
    show (RecursiveDefinitionNonFunction e) = "Evaluation Error - Trying do a recursive definition here, " ++ show e ++ " for a non-function type is not allowed."
    show (TypeError t e) = "Evaluation Error - When trying to determine the type of " ++ show e ++ 
                                   " to determine how to do equality, a type error occured. The error was: " ++ show t

free_variables :: LambdaExpression -> Set String
free_variables (Variable x) = Set.singleton x
free_variables (Abstraction (x, _) y) = Set.delete x $ free_variables y
free_variables (BinaryOper x _ y) = on Set.union free_variables x y
free_variables (UnaryOper _ x) = free_variables x
free_variables (If x y z) = Set.unions (map free_variables [x,y,z])
free_variables _ = Set.empty

isValue :: LambdaExpression -> Bool
isValue (Abstraction _ _) = True
isValue (Number _) = True
isValue (Boolean _) = True
isValue Nil = True
isValue (Cons (Number _) tl) = isValue tl
isValue _ = False

is_list_int :: LambdaExpression -> Bool
is_list_int Nil = True
is_list_int (Cons (Number _) tl) = is_list_int tl
is_list_int _ = False

functionType :: Type -> Bool
functionType (TFun _ _) = True
functionType _ = False

y_cbv :: LambdaExpression
y_cbv = Abstraction ("f", TFake) (BinaryOper 
    (Abstraction ("x", TFake) (BinaryOper (Variable "f") Application (Abstraction ("v", TFake) (BinaryOper (BinaryOper (Variable "x") Application (Variable "x")) Application (Variable "v")))))
    Application
    (Abstraction ("x", TFake) (BinaryOper (Variable "f") Application (Abstraction ("v", TFake) (BinaryOper (BinaryOper (Variable "x") Application (Variable "x")) Application (Variable "v"))))))

extract_num_op :: BinaryOp -> Integer -> Integer -> Integer
extract_num_op Plus = (+)
extract_num_op Minus = (-)
extract_num_op Times = (*)
extract_num_op _ = error "This is only intended to be used on the arithmetic operations (except divide)."

extract_bool_op :: BinaryOp -> Bool -> Bool -> Bool
extract_bool_op And = (&&)
extract_bool_op Or = (||)
extract_bool_op _ = error "This is only intended to be used on boolean operations."

evalCBV :: LambdaExpression -> Either EvaluationError LambdaExpression
evalCBV (Variable var) = Left $ UndefinedVariableE var
evalCBV (If x y z) = do x' <- evalCBV x
                        case x' of
                            Boolean True -> evalCBV y
                            Boolean False -> evalCBV z
                            _ -> Left $ TypeMismatchE TBool x
evalCBV (BinaryOper v Application e) | isValue e && isValue v = case v of 
                                                                    Abstraction (var, _) expr -> evalCBV (substitute expr var e)
                                                                    v' -> Left $ ExpectedFunctionE v'
                                     | isValue v = flip BinaryOper Application v <$> evalCBV e  >>= evalCBV
                                     | otherwise = flip BinaryOper Application <$> evalCBV v <*> pure e >>= evalCBV
evalCBV e@(BinaryOper x op y) | op `elem` [Plus, Minus, Times] = do x' <- evalCBV x
                                                                    y' <- evalCBV y
                                                                    case (x', y') of
                                                                      (Number a, Number b) -> pure $ Number (extract_num_op op a b)
                                                                      (Number _, b) -> Left $ TypeMismatchE TInt b
                                                                      (a, _) -> Left $ TypeMismatchE TInt a
                              | op == Divide = do x' <- evalCBV x
                                                  y' <- evalCBV y
                                                  case (x', y') of
                                                      (Number _, Number 0) -> Left $ DivideByZeroE e
                                                      (Number a, Number b) -> pure $ Number (a `div` b)
                                                      (Number _, b) -> Left $ TypeMismatchE TInt b
                                                      (a, _) -> Left $ TypeMismatchE TInt a

                              | op `elem` [And, Or] = do x' <- evalCBV x
                                                         y' <- evalCBV y
                                                         case (x', y') of
                                                              (Boolean a, Boolean b) -> pure $ Boolean (extract_bool_op op a b)
                                                              (Boolean _, b) -> Left $ TypeMismatchE TBool b
                                                              (a, _) -> Left $ TypeMismatchE TBool a
                              | op == Equal = do x' <- evalCBV x
                                                 y' <- evalCBV y
                                                 tx <- either (Left . flip TypeError x') pure (typeOf Map.empty x')
                                                 ty <- either (Left . flip TypeError y') pure (typeOf Map.empty y')
                                                 if functionType tx
                                                    then Left $ NoEqualityForFunctionsE x
                                                    else if tx == ty 
                                                            then pure (Boolean (x' == y'))
                                                            else Left $ TypeMismatchE tx y
                              | otherwise = do x' <- evalCBV x
                                               y' <- evalCBV y
                                               case (x', y') of
                                                    (Number a, Number b) -> pure $ Boolean (a < b)
                                                    (Number _, b) -> Left $ TypeMismatchE TInt b
                                                    (a, _) -> Left $ TypeMismatchE TInt a
evalCBV (UnaryOper Neg x) = do x' <- evalCBV x
                               case x' of
                                    Number n -> pure $ Number (negate n)
                                    a -> Left $ TypeMismatchE TInt a
evalCBV (UnaryOper Not x) = do x' <- evalCBV x
                               case x' of
                                    Boolean b -> pure $ Boolean (not b)
                                    a -> Left $ TypeMismatchE TBool a
evalCBV x@(Let var y z) = if var `Set.member` free_variables y then Left $ RecursiveDefinitionLet x else evalCBV (BinaryOper (Abstraction (var, TFake) z) Application y)
evalCBV x@(LetRec (var,t) y z) = case t of
                                    TFun _ _ -> evalCBV (BinaryOper (Abstraction (var,t) z) Application (BinaryOper y_cbv Application (Abstraction (var, t) y)))
                                    _ -> if var `Set.member` free_variables y then Left $ RecursiveDefinitionNonFunction x else evalCBV (BinaryOper (Abstraction (var, t) z) Application y)
evalCBV x@(Cons (Number n) tl) = if is_list_int tl 
                                    then pure x 
                                    else do tl' <- evalCBV tl
                                            if is_list_int tl' 
                                              then pure $ Cons (Number n) tl'
                                              else Left $ TypeMismatchE TIntList tl
evalCBV (Cons h tl) = do h' <- evalCBV h
                         case h' of
                            n@(Number _) -> Cons n <$> evalCBV tl
                            _ -> Left $ TypeMismatchE TIntList h
evalCBV (Match ls e1 e2) | is_list_int ls = case ls of
                                              Nil -> evalCBV e1
                                              Cons h t -> evalCBV (BinaryOper (BinaryOper e2 Application h) Application t)
                                              _ -> Left $ TypeMismatchE TIntList ls
                         | otherwise = do ls' <- evalCBV ls
                                          if is_list_int ls' 
                                            then evalCBV (Match ls' e1 e2)
                                            else Left $ TypeMismatchE TIntList ls
evalCBV val = pure val