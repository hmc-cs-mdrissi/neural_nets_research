module LambdaInterp where

import LambdaParser
import Data.Function
import Data.Set (Set)
import qualified Data.Set as Set

data EvaluationError = UnboundVariable String | DivideByZero | NotImplemented | TypeError String

instance Show EvaluationError where
    show (UnboundVariable var) = "An unbound variable, " ++ var ++ " was found."
    show DivideByZero = "A division by zero."
    show NotImplemented = "This feature has not yet been implemented."
    show (TypeError message) = "Wrong Type! " ++ message

free_variables :: LambdaExpression -> Set String
free_variables (Variable x) = Set.singleton x
free_variables (Application x y) = on Set.union free_variables x y
free_variables (Abstraction x y) = Set.delete x $ free_variables y
free_variables (Cons x y) = on Set.union free_variables x y
free_variables (Foldr x y z) = (on Set.union free_variables x y) `Set.union` free_variables z
free_variables (NatBinOp _ x y) = on Set.union free_variables x y
free_variables _ = Set.empty

check_nat_list :: LambdaExpression -> Bool
check_nat_list Nil = True
check_nat_list (Cons (Natural _) l) = check_nat_list l
check_nat_list _ = False

check_value :: LambdaExpression -> Bool
check_value (Abstraction _ _) = True
check_value (Natural _) = True
check_value x = check_nat_list x

extract_binary_op :: NatBinOp -> (Integer -> Integer -> Integer)
extract_binary_op Plus = (+)
extract_binary_op Minus = (-)
extract_binary_op Div = div
extract_binary_op Mult = (*)

evalCBV :: LambdaExpression -> Either EvaluationError LambdaExpression
evalCBV n@(Natural _) = Right n
evalCBV Nil = Right Nil
evalCBV v@(Abstraction _ _) = Right v
evalCBV v@(Cons n@(Natural _) e)  = if check_nat_list e then 
                                        Right v 
                                    else 
                                        case evalCBV e of 
                                            Right Nil -> Right $ Cons n Nil
                                            Right v@(Cons _ _) -> Cons n <$> evalCBV v
                                            Right _ -> Left $ TypeError "The second argument to a cons should be a natlist."
                                            Left e -> Left e
evalCBV (Cons e1 e2)= case evalCBV e1 of
                        Right v@(Natural _) -> Cons v <$> (evalCBV e2)
                        Right _ -> Left $ TypeError "Non-nat list head"
                        Left e -> Left e
evalCBV (Application v@(Abstraction var expr) e) | check_value e = evalCBV (substitute expr var e)
                                                 | otherwise = Application v <$> evalCBV e >>= evalCBV
evalCBV (Application e1 e2) = case evalCBV e1 of
                                Right v@(Abstraction _ _) -> Application v <$> (evalCBV e2) >>= evalCBV
                                Right _ -> Left $ TypeError "Applying non-function"
                                Left e -> Left e
evalCBV (Variable var) = Left $ UnboundVariable $ var
evalCBV (NatBinOp op (Natural e1) (Natural e2)) = Right $ Natural ((extract_binary_op op) e1 e2)
evalCBV (NatBinOp op (Natural e1) e2) = case evalCBV e2 of
                                            Right n@(Natural _) -> evalCBV (NatBinOp op (Natural e1) n)
                                            Right _ -> Left $ TypeError "Second argument to a natural binary operator should be a natural."
                                            Left e -> Left e
evalCBV (NatBinOp op e1 e2) = case evalCBV e1 of
                                Right n@(Natural _) -> evalCBV $ NatBinOp op n e2
                                Right _ -> Left $ TypeError "First argument to a natural binary operator should be a natural."
                                Left e -> Left e
evalCBV (Foldr fun base Nil) = evalCBV base
evalCBV (Foldr fun@(Abstraction _ _) base list@(Cons head tail)) | check_nat_list list = evalCBV $ Application (Application fun head) (Foldr fun base tail)
                                                                 | otherwise = Foldr fun base <$> evalCBV list >>= evalCBV
evalCBV (Foldr fun@(Abstraction _ _) base list) = case evalCBV list of
                                                    Right Nil -> evalCBV base
                                                    Right list'@(Cons _ _) -> evalCBV $ Foldr fun base list'
                                                    Right _ -> Left $ TypeError "Bad list"
                                                    Left e -> Left e
evalCBV (Foldr fun base list) = case evalCBV fun of
                                  Right fun'@(Abstraction _ _) -> evalCBV $ Foldr fun' base list
                                  Right _ -> Left $ TypeError "Bad fun"
                                  Left e -> Left e