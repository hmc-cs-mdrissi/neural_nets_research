module LambdaInterp where

import LambdaParser
import Data.Function
import Data.Set (Set)
import qualified Data.Set as Set

import Debug.Trace

data EvaluationError = UnboundVariable String | DivideByZero | NotImplemented

instance Show EvaluationError where
    show (UnboundVariable var) = "An unbound variable, " ++ var ++ " was found."
    show DivideByZero = "A division by zero."
    show NotImplemented = "This feature has not yet been implemented."

free_variables :: LambdaExpression -> Set String
free_variables (Variable x) = Set.singleton x
free_variables (Application x y) = on Set.union free_variables x y
free_variables (Abstraction x y) = Set.delete x $ free_variables y
free_variables (Cons x y) = on Set.union free_variables x y
free_variables (Foldr x y z) = (on Set.union free_variables x y) `Set.union` free_variables z
free_variables (NatBinOp _ x y) = on Set.union free_variables x y
free_variables _ = Set.empty

bound_variables :: LambdaExpression -> Set String
bound_variables (Variable x) = Set.empty
bound_variables (Application x y) = on Set.union bound_variables x y
bound_variables (Abstraction x y) = Set.insert x $ bound_variables y
bound_variables (Cons x y) = on Set.union bound_variables x y
bound_variables (Foldr x y z) = (on Set.union bound_variables x y) `Set.union` bound_variables z
bound_variables (NatBinOp _ x y) = on Set.union bound_variables x y
bound_variables _ = Set.empty

fresh_variable :: Set String -> String
fresh_variable bound = fresh_variable_aux bound 1
    where fresh_variable_aux bound n = if (name n) `Set.notMember` bound then name n else fresh_variable_aux bound (n+1)
          name n = "a" ++ show n

check_nat_list :: LambdaExpression -> Bool
check_nat_list Nil = True
check_nat_list (Cons (Natural _) l) = check_nat_list l
check_nat_list _ = False

check_value :: LambdaExpression -> Bool
check_value (Abstraction _ _) = True
check_value (Natural _) = True
check_value x = check_nat_list x

extract_binary_op :: NatBinOp -> (Int -> Int -> Int)
extract_binary_op Plus = (+)
extract_binary_op Minus = (-)
extract_binary_op Div = div
extract_binary_op Mult = (*)

evalCBV :: LambdaExpression -> Either EvaluationError LambdaExpression
evalCBV n@(Natural _) = Right n
evalCBV Nil = Right Nil
evalCBV v@(Abstraction _ _) = Right v
evalCBV v@(Cons n@(Natural _) e)  = if check_nat_list e then Right v else Cons n <$> evalCBV e >>= evalCBV
evalCBV (Cons e1 e2) = Cons <$> evalCBV e1 <*> pure e2 >>= evalCBV
evalCBV (Application v@(Abstraction var expr) e) | check_value e = evalCBV (substitute expr var e)
                                                 | otherwise = Application v <$> evalCBV e >>= evalCBV
evalCBV (Application e1 e2) = Application <$> evalCBV e1 <*> pure e2 >>= evalCBV
evalCBV (Variable var) = Left $ UnboundVariable $ var
evalCBV _ = Left $ NotImplemented

--     Foldr LambdaExpression LambdaExpression LambdaExpression | 
--     NatBinOp NatBinOp LambdaExpression LambdaExpression
--     evalCBV_with_count :: Int -> LambdaExpression -> Either (String, Int) EvaluationError
